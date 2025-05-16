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

def new_point_with_retinal_image_invariance(u, v, X, Y, d, listing_angle_deg=17.0):
    """
    視線 (X, Y, d) の変化に伴い、網膜像を Listing の法則に基づいて回転補正。
    u, v: 元の画像座標
    X, Y: 視線ベクトルのx, y成分（Zはdとする）
    d: 視線のZ成分（通常、視線ベクトルのZ方向）
    """
    # ① Listingの法則による torsion（網膜像の回転角）を計算
    theta_h = math.atan2(X, d)
    theta_v = math.atan2(Y, d)
    listing_rad = math.radians(listing_angle_deg)
    torsion_rad = -math.tan(theta_h) * math.tan(theta_v) * math.sin(listing_rad)

    # ② 回転行列を使って (u, v) を回転補正
    cos_t = math.cos(-torsion_rad)  # -torsion_rad で網膜像に逆回転をかける
    sin_t = math.sin(-torsion_rad)
    u_rot = cos_t * u - sin_t * v
    v_rot = sin_t * u + cos_t * v

    # ③ 射影変換（画像変形の本質的部分）
    A = math.sqrt(X*X + d*d)
    B = math.sqrt(X*X + Y*Y + d*d)
    denominator = (X*u_rot*B - d*d*A + Y*d*v_rot)

    if abs(denominator) < 1e-5:  # 零除算回避
        return u, v  # 回転前の座標を返す

    new_u = -d * (d*u_rot*B + X*d*A - X*Y*v_rot) / denominator
    new_v = -d * (v_rot*A*A + Y*d*A) / denominator

    return new_u, new_v


# ------------------ スムージング初期化 ------------------
R = 0.45
last_valid_gaze = None
last_M = None
last_torsion_deg = None
last_gaze = np.array([0, 0, 1])  # 正面を初期値とする
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

        dst = cv2.warpPerspective(img, last_M, (window_width, window_height))
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