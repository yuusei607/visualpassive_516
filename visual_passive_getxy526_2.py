
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
screen_width = 1920
screen_height = 1080
window_width = screen_width
window_height = screen_height

# ------------------ 画像読み込み ------------------
img = cv2.imread(image_path)
if img is None:
    print("Failed to load image.")
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
    if denominator < 1e-5:  # 零除算回避
        if abs(denominator)<1e-5:
              return u, v
        
    new_u = -d * (d*u*B + X*d*A - X*Y*v) / denominator
    new_v = -d * (v*A*A + Y*d*A) / denominator

    

    torsion = math.atan2(Y,X)
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
    
    torsion = -0.3 * torsion

    n1 = X/B
    n2 = Y/B
    n3 = d/B
    cos = math.cos(torsion)
    cos_ = 1 - cos
    sin = math.sin(torsion)
    R = np.array([[n1*n1*cos_ + cos, n1*n2*cos_ - n3*sin, n1*n3*cos_ + n2*sin],
                  [n1*n2*cos_ + n3*sin, n2*n2*cos_ + cos, n2*n3*cos_ - n1*sin],
                  [n1*n3*cos_ - n2*sin, n2*n3*cos_ + n1*sin, n3*n3*cos_ + cos]])
    
    new = R@np.array([[new_u],[new_v],[d]])
    
    new_u2 = new[0]*d/new[2]
    new_v2 = new[1]*d/new[2]
    return new_u2, new_v2
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
R = 1.0
last_valid_gaze = None
last_M = None
last_gaze_centered = None
start = time.time()
frame_count = 0

gaze_history = deque(maxlen=5)
# 指数移動平均用
alpha = 0.60
alpha_bad = 0.04
gaze_avg = None  # 初期値
size = 1.0
#縦横比
ratio_h = 0.6
ratio_w = 0.4
# ------------------ メインループ ------------------
while True:
    # now = time.time()
    # time_lapse = now - start
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

    gaze_centered1 = np.array([gaze_screen[0] - screen_width / 2, gaze_screen[1] - screen_height / 2])
    #本来、画面の端を[1,1]としたいが、このままではworld cameraの端が[1,1]となってしまうのでここで補正
    gaze_centered1 = (np.array([[ratio_w,0],[0,ratio_h]])@gaze_centered1.T).T
    gaze_centered = tuple(gaze_centered1)
    # --- 画像補正処理 ---
    pts1_original = np.float32([
        [0, 0],
        [image_width, 0],
        [0, image_height],
        [image_width, image_height]
    ])

    image_height_in_mm = 1200
    k2 = image_height_in_mm / image_height #k2はピクセルからmmへの変換スケール(読み込んだimageのピクセル→自分で入力したimageの高さ)
    pts1_in_mm = (size) * k2 * np.float32([
        [-image_width/2, -image_height/2],
        [image_width/2, -image_height/2],
        [-image_width/2, image_height/2],
        [image_width/2, image_height/2]
    ])

    screen_height_in_mm = 1400
    screen_distance_in_mm = 500
    k = screen_height_in_mm / screen_height #kは最初に入力したスクリーンの画素数からここで設定したscreen_heightの高さ(mm)への変換スケール

    gaze_in_mm = np.array([k * gaze_centered[0], k * gaze_centered[1], screen_distance_in_mm])

    x = 0
    y = 0
    par = np.float32([x, y])

    #if np.linalg.norm(gaze - gaze_avg) < 0.3:
    if confidence > 0.5:
        if last_gaze_centered is None or np.linalg.norm(np.array(gaze_centered) - np.array(last_gaze_centered)) > 10:
            last_gaze_centered = gaze_centered
            pts2_in_mm = np.float32([
                list(new_point_with_retinal_image_invariance(*(pts1_in_mm[0] + par), *gaze_in_mm)),
                list(new_point_with_retinal_image_invariance(*(pts1_in_mm[1] + par), *gaze_in_mm)),
                list(new_point_with_retinal_image_invariance(*(pts1_in_mm[2] + par), *gaze_in_mm)),
                list(new_point_with_retinal_image_invariance(*(pts1_in_mm[3] + par), *gaze_in_mm)),
            ])
            

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

    time.sleep(0.002)

cv2.destroyAllWindows()
