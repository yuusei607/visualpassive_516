import time
import zmq
import msgpack
import numpy as np
import cv2
import argparse
import os
import math
from collections import deque
pi = np.pi
# ------------------ 引数処理 ------------------
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, required=True, help="Path to image file")
args = parser.parse_args()

image_path = args.image
if not os.path.exists(image_path):
    print(f"Image file not found: {image_path}")
    exit(1)

# ------------------ Pupil Core 接続 ------------------
ctx = zmq.Context()
ip = 'localhost'
port = 50020

pupil_remote = ctx.socket(zmq.REQ)
pupil_remote.connect(f"tcp://{ip}:{port}")

pupil_remote.send_string('SUB_PORT')
sub_port = pupil_remote.recv_string()

subscriber = ctx.socket(zmq.SUB)
subscriber.connect(f"tcp://{ip}:{sub_port}")
subscriber.subscribe('gaze.')

# ------------------ ディスプレイ設定 ------------------
screen_width = 3840//5
screen_height = 2160//5
window_width = screen_width
window_height = screen_height

# ------------------ 画像読み込み ------------------
cap = cv2.VideoCapture(image_path)
if not cap.isOpened():
    print("Failed to load video.")
    exit(1)
ret, img = cap.read()
if not ret:
    print("Failed to read the first frame.")
    exit(1)

image_height, image_width, ch = img.shape

cv2.namedWindow("Gaze Viewer", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Gaze Viewer", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ------------------ 補助関数 ------------------
def scale_translate(u, v, cX, cY, s):
    return (s*u - cX), (s*v - cY)

def new_point_with_retinal_image_invariance(u, v, X, Y, d):
    A = math.sqrt(X*X + d*d)
    B = math.sqrt(X*X + Y*Y + d*d)
    denominator = (X*u*B - d*d*A + Y*d*v)
    if abs(denominator) < 1e-5:  # 零除算回避
        return u, v
    new_u = -d * (d*u*B + X*d*A - X*Y*v) / denominator
    new_v = -d * (v*A*A + Y*d*A) / denominator
    return new_u, new_v
def get_torsion_angle_deg(gaze_mm):
    """
    視線ベクトル gaze_mm に基づいて、網膜像が回転する角度（degree）を計算。
    通常、右斜め上を見ると反時計回りのトーションが生じる。
    """
    x, y, z = gaze_mm
    torsion = math.atan2(y, x)  # 単純な水平面上の角度
    if torsion < 0:
        torsion += 2*pi

 
    if pi/4<=torsion<pi*3/4:
        torsion = pi/2 - torsion
    elif pi*3/4<=torsion<5*pi/4:
        torsion = torsion - pi
    elif 5*pi/4<=torsion<=7*pi/4:
        torsion = 3*pi/2 - torsion
    elif 7*pi/4<=torsion<=2*pi:
        torsion = 2*pi - torsion
    elif 0<=torsion<pi/4:
        torsion = -torsion

    return np.degrees(torsion) * (0.15)  # スケーリング係数（調整可能）


# ------------------ スムージング初期化 ------------------
R = 0.45
last_valid_gaze = None
last_M = None
last_gaze_centered = None
start = time.time()
frame_count = 0

gaze_history = deque(maxlen=5)
# 指数移動平均用
alpha = 0.85
alpha_bad = 0.04
gaze_avg = None  # 初期値
size = 1.5
# ------------------ メインループ ------------------
while True:
    # now = time.time()
    # time_lapse = now - start
    ret, img = cap.read()
    if not ret:
        print("Failed to read the first frame.")
        exit(1)
    frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    # 最新の視線取得
    norm_pos = None
    while True:
        try:
            topic, payload = subscriber.recv_multipart(flags=zmq.NOBLOCK)
            message = msgpack.loads(payload, raw=False)
            norm_pos = message.get("norm_pos")
            confidence = message.get("confidence", 0.0)  # デフォルト0.0にしておく

        except zmq.Again:
            break

    if norm_pos:
        gaze = np.array([norm_pos[0], 1 - norm_pos[1]])  # Y軸反転
        last_valid_gaze = gaze
    elif last_valid_gaze is not None:
        gaze = last_valid_gaze
    else:
        gaze = None

    gaze_history.append(gaze)

    if gaze is None:
        dst = img.copy()
        cv2.imshow("Gaze Viewer", dst)
        if cv2.waitKey(1) == ord('q'):
            break
        time.sleep(0.002)
        continue

    if gaze_avg is None:
        gaze_avg = gaze  # 初回のみ初期化
    elif confidence > 0.3:
        gaze_avg = (1 - alpha) * gaze_avg + alpha * gaze  # 指数移動平均
        #gaze_avg = np.mean(gaze_history,axis = 0)
    else:
        gaze_avg = (1 - alpha_bad) * gaze_avg + alpha_bad * gaze


    gaze_screen = (screen_width * gaze_avg[0], screen_height * gaze_avg[1])
    gaze_centered = gaze_screen[0] - screen_width / 2, gaze_screen[1] - screen_height / 2

    # --- 画像補正処理 ---
    pts1_original = np.float32([
        [0, 0],
        [image_width, 0],
        [0, image_height],
        [image_width, image_height]
    ])

    image_height_in_mm = 500
    k2 = image_height_in_mm / image_height
    pts1_in_mm = (size/R) * k2 * np.float32([
        [-image_width/2, -image_height/2],
        [image_width/2, -image_height/2],
        [-image_width/2, image_height/2],
        [image_width/2, image_height/2]
    ])

    screen_height_in_mm = 500
    screen_distance_in_mm = 300
    k = screen_height_in_mm / screen_height

    gaze_in_mm = np.array([k * gaze_centered[0], k * gaze_centered[1], screen_distance_in_mm])
    #frame_count += 1
    # x = min(frame_count,0)
    # y = min(int(frame_count*0.25),0)
    x = 0
    y = 50
    par = np.float32([x, y])

    #if np.linalg.norm(gaze - gaze_avg) < 0.3:
    if confidence > 0.7:
        if last_gaze_centered is None or np.linalg.norm(np.array(gaze_centered) - np.array(last_gaze_centered)) > 10:
            last_gaze_centered = gaze_centered
            # --- 回転行列を作成 ---
            torsion_deg = get_torsion_angle_deg(gaze_in_mm)
            center = (image_width // 2, image_height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, torsion_deg, 1.0)
            rotated_img = cv2.warpAffine(img, rotation_matrix, (image_width, image_height))
            pts2_in_mm = np.float32([
                list(new_point_with_retinal_image_invariance(*(pts1_in_mm[0] + par), *gaze_in_mm)),
                list(new_point_with_retinal_image_invariance(*(pts1_in_mm[1] + par), *gaze_in_mm)),
                list(new_point_with_retinal_image_invariance(*(pts1_in_mm[2] + par), *gaze_in_mm)),
                list(new_point_with_retinal_image_invariance(*(pts1_in_mm[3] + par), *gaze_in_mm)),
            ])
            B = np.array([[R, 0],
                          [0, R]])
            x_parallel = 0
            y_parallel = 0
            v2 = np.array([[x_parallel]*4, [y_parallel]*4])

            pts2_in_mm_t = B @ (pts2_in_mm.T + v2)
            pts2_in_mm = pts2_in_mm_t.T

            pts2_display = []
            for pt in pts2_in_mm:
                u, v_mm = scale_translate(pt[0], pt[1], -window_width / 2, -window_height / 2, 1.0 / k)
                pts2_display.append([u, v_mm])

            pts2_display = np.float32(pts2_display)
            last_M = cv2.getPerspectiveTransform(pts1_original, pts2_display)

        dst = cv2.warpPerspective(rotated_img, last_M, (window_width, window_height))
    else:
        if last_M is not None:
            dst = cv2.warpPerspective(img, last_M, (window_width, window_height))
        else:
            dst = img.copy()

    cv2.imshow("Gaze Viewer", dst)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    time.sleep(0.001)

cv2.destroyAllWindows()