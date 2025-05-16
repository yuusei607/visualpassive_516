import time
import zmq
import msgpack
import numpy as np
import cv2
import argparse
import os
import math
from collections import deque

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
    if abs(denominator) < 1e-5:  # 零除算回避
        return u, v
    new_u = -d * (d*u*B + X*d*A - X*Y*v) / denominator
    new_v = -d * (v*A*A + Y*d*A) / denominator
    return new_u, new_v

# ------------------ スムージング初期化 ------------------
gaze_history = deque(maxlen=5)
R = 1.0
last_valid_gaze = None
last_M = None
last_gaze_centered = None
start = time.time()
frame_count = 0

# ------------------ メインループ ------------------
while True:
    now = time.time()
    frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    # 最新の視線取得
    norm_pos = None
    while True:
        try:
            topic, payload = subscriber.recv_multipart(flags=zmq.NOBLOCK)
            message = msgpack.loads(payload, raw=False)
            norm_pos = message.get("norm_pos")
        except zmq.Again:
            break

    if norm_pos:
        gaze = np.array([norm_pos[0], 1 - norm_pos[1]])  # Y軸反転
        last_valid_gaze = gaze
        gaze_history.append(gaze)
    elif last_valid_gaze is not None:
        gaze_history.append(last_valid_gaze)

    if len(gaze_history) == 0:
        dst = img.copy()
        cv2.imshow("Gaze Viewer", dst)
        if cv2.waitKey(1) == ord('q'):
            break
        time.sleep(0.002)
        continue

    gaze_avg = np.mean(gaze_history, axis=0)
    gaze_screen = (screen_width * gaze_avg[0], screen_height * gaze_avg[1])
    gaze_centered = gaze_screen[0] - screen_width / 2, gaze_screen[1] - screen_height / 2

    # --- 射影変換行列の再利用と条件付き再計算 ---
    if np.linalg.norm(gaze_centered) < 1500:#元は900
        if last_gaze_centered is None or np.linalg.norm(np.array(gaze_centered) - np.array(last_gaze_centered)) > 30:#元は10
            last_gaze_centered = gaze_centered

            pts1_original = np.float32([
                [0, 0],
                [image_width, 0],
                [0, image_height],
                [image_width, image_height]
            ])

            image_height_in_mm = 500
            k2 = image_height_in_mm / image_height
            pts1_in_mm = (3.8/R) * k2 * np.float32([
                [-image_width/2, -image_height/2],
                [image_width/2, -image_height/2],
                [-image_width/2, image_height/2],
                [image_width/2, image_height/2]
            ])

            screen_height_in_mm = 400
            screen_distance_in_mm = 300
            k = screen_height_in_mm / screen_height

            gaze_in_mm = np.array([k * gaze_centered[0], k * gaze_centered[1], screen_distance_in_mm])

            frame_count += 1
            x = min(frame_count, 800)
            y = min(int(frame_count*0.25),200)
            par = np.float32([x, 0])

            pts2_in_mm = np.float32([
                list(new_point_with_retinal_image_invariance(*(pts1_in_mm[0] + par), *gaze_in_mm)),
                list(new_point_with_retinal_image_invariance(*(pts1_in_mm[1] + par), *gaze_in_mm)),
                list(new_point_with_retinal_image_invariance(*(pts1_in_mm[2] + par), *gaze_in_mm)),
                list(new_point_with_retinal_image_invariance(*(pts1_in_mm[3] + par), *gaze_in_mm)),
            ])

            B = np.array([[R, 0], [0, R]])
            v2 = np.array([[0]*4, [0]*4])  # 並進なし

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
        dst = img.copy()

    cv2.imshow("Gaze Viewer", dst)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    time.sleep(0.002)

cv2.destroyAllWindows()





