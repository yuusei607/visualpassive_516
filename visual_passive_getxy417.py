import time
import zmq
import msgpack
import numpy as np
import cv2
import argparse
import os
import math

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
screen_width = 1440
screen_height = 900
window_width = screen_width
window_height = screen_height

# ------------------ 画像読み込み ------------------
img = cv2.imread(image_path)
if img is None:
    print("Failed to load image.")
    exit(1)

image_height, image_width, ch = img.shape

# ------------------ 初期化 ------------------
cv2.namedWindow("Gaze Viewer", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Gaze Viewer", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

gaze_normalized = np.zeros((2))

# ------------------ 補助関数 ------------------
def scale_translate(u, v, cX, cY, s):
    return (s*u - cX), (s*v - cY)

def new_point_with_retinal_image_invariance(u, v, X, Y, d):
    A = math.sqrt(X*X + d*d)
    B = math.sqrt(X*X + Y*Y + d*d)
    return -d * (d*u*B + X*d*A - X*Y*v) / (X*u*B - d*d*A + Y*d*v), -d * (v*A*A + Y*d*A) / (X*u*B - d*d*A + Y*d*v)

# ------------------ メインループ ------------------
while True:
    frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    norm_pos = None
    while True:
        try:
            topic, payload = subscriber.recv_multipart(flags=zmq.NOBLOCK)
            message = msgpack.loads(payload, raw=False)
            norm_pos = message.get("norm_pos")
        except zmq.Again:
            break

    if norm_pos:
        gaze_normalized = np.array([norm_pos[0], 1 - norm_pos[1]])  # Flip Y axis
        gaze_screen = (screen_width * gaze_normalized[0], screen_height * gaze_normalized[1])
        gaze = gaze_screen[0] - screen_width/2, gaze_screen[1] - screen_height/2
        

        pts1_original = np.float32([
            [0, 0],
            [image_width, 0],
            [0, image_height],
            [image_width, image_height]
        ])

        image_height_in_mm = 500
        k2 = image_height_in_mm / image_height

        pts1_in_mm = k2 * np.float32([
            [-image_width/2, -image_height/2],
            [image_width/2, -image_height/2],
            [-image_width/2, image_height/2],
            [image_width/2, image_height/2]
        ])

        screen_height_in_mm = 1180
        screen_distance_in_mm = 750
        k = screen_height_in_mm / screen_height

        gaze_in_mm = np.array([k * gaze[0], k * gaze[1], screen_distance_in_mm])

        pts2_in_mm = np.float32([
            list(new_point_with_retinal_image_invariance(*pts1_in_mm[0], *gaze_in_mm)),
            list(new_point_with_retinal_image_invariance(*pts1_in_mm[1], *gaze_in_mm)),
            list(new_point_with_retinal_image_invariance(*pts1_in_mm[2], *gaze_in_mm)),
            list(new_point_with_retinal_image_invariance(*pts1_in_mm[3], *gaze_in_mm)),
        ])

        A = np.array([[1.0, 0.0],   # ここを変えると拡大・回転・せん断ができる
              [0.0, 1.0]])
        v = np.array([0, 0])     # 右に20px、上に10pxずらす例

            # 2. 元のpts2_in_mm（float32の4点）を使って変換適用
        pts2_display = []
        for pt in pts2_in_mm:
            u, v_mm = scale_translate(pt[0], pt[1], -window_width / 2, -window_height / 2, 1.0 / k)
            transformed = A @ (np.array([u, v_mm]) + v)
            pts2_display.append(transformed)

        pts2_display = np.float32(pts2_display)

        M = cv2.getPerspectiveTransform(pts1_original, pts2_display)
        dst = cv2.warpPerspective(img, M, (window_width, window_height))
    else:
        dst = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

    cv2.imshow("Gaze Viewer", dst)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()