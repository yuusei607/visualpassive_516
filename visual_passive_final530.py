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
flag1 = 0
flag2 = 0
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
    #この下の二式は、本来ロドリゲスの回転公式は右手系において用いる必要があるが、
    #現状の座標の取りかたでは左手系になってしまうのでY軸を反転させることで無理やり右手系にしている
    #したがって、最後の出力でもreturn x, yではなく、もとの座標に戻すためにreturn x, -yとしてyの上下を再び反転している
    v = -v 
    Y = -Y

    A = math.sqrt(X*X + d*d)
    B = math.sqrt(X*X + Y*Y + d*d)
    denominator = (X*u*B - d*d*A + Y*d*v)
    if denominator < 1e-5:  # 零除算回避
        if abs(denominator)<1e-5:
              return u, v

    new_u = -d * (d*u*B + X*d*A - X*Y*v) / denominator
    new_v = -d * (v*A*A + Y*d*A) / denominator



    torsion = math.atan2(-Y,X)
    if torsion < 0:
        torsion += 2*pi


    if pi/4<=torsion<pi*3/4:
        torsion = torsion - pi/2
    elif pi*3/4<=torsion<5*pi/4:
        torsion = pi - torsion
    elif 5*pi/4<=torsion<=7*pi/4:
        torsion = torsion - 3*pi/2
    elif 7*pi/4<=torsion<=2*pi:
        torsion = 2*pi - torsion
    elif 0<=torsion<pi/4:
        torsion = -torsion
    #もし中心からl[cm]離れたら倍率a*torsion分回転する。
    #中心から十分離れていないのに回転を与えてしまうと中心付近で小さな変位に対して
    #回転角が大きくなりすぎるのでこうした設定にした。
    #これらは状況に応じて変えていい値
    l = 300
    a = 0.001
    if A > l:
        torsion = -a*(A-l)*torsion
    else:
        torsion = 0

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


    return new_u2, -new_v2

#これはロドリゲスで変換した後、そのZ座標が負になっているものを炙り出すための関数
def find_turning_minus(u, v, X, Y, d):
    
    v = -v
    Y = -Y
    A = math.sqrt(X*X + d*d)
    B = math.sqrt(X*X + Y*Y + d*d)
    sintheta = X/A
    costheta = d/A
    sinphi = Y/B
    cosphi = A/B
    dd = -sintheta*u - costheta*sinphi*v + costheta*cosphi*d
    return dd



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
ratio_h = 0.32
ratio_w = 0.28
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
        #裏返りを消す
        sign = np.float32([find_turning_minus(*pts1_in_mm[0], *gaze_in_mm),
                                  find_turning_minus(*pts1_in_mm[1], *gaze_in_mm),
                                  find_turning_minus(*pts1_in_mm[2], *gaze_in_mm),
                                  find_turning_minus(*pts1_in_mm[3], *gaze_in_mm)])
        #一回だけX,Yを取得したい(毎回取得し直すと重い)のでflagを使う
        if flag1 == 0:
            height, width = dst.shape[:2]
            X, Y = np.meshgrid(np.arange(width), np.arange(height))
            flag1 += 1
        #以下においてsign[i]>0はp[i]の点が裏返ってないことを表す
        #画像の左上と右上が裏返ってないならこの二点よりも上側の領域をmaskする(黒にする)
        if sign[0] > 0 and sign[1] > 0:
            mask = Y < min(pts2_display[0][1],pts2_display[1][1])
            dst[mask] = (0, 0, 0)
        #画像の左上と左下の点が裏返ってないならこの二点よりも左の領域をmaskする
        if sign[0] > 0 and sign[2] > 0:
            mask = (pts2_display[0][0]-pts2_display[2][0])*(Y-pts2_display[0][1]) < (pts2_display[0][1]-pts2_display[2][1])*(X-pts2_display[0][0])
            dst[mask] = (0, 0, 0)
        #画像の右上と右下が裏返ってないならこの二点よりも右側の領域をmaskする
        if sign[1] > 0 and sign[3] > 0:
            mask = (pts2_display[1][0]-pts2_display[3][0])*(Y-pts2_display[1][1]) > (pts2_display[1][1]-pts2_display[3][1])*(X-pts2_display[1][0])
            dst[mask] = (0, 0, 0)
        #画像の右下と左下の点が裏返ってないならこの二点よりも下側の領域をmaskする
        if sign[2] > 0 and sign[3] > 0:
            mask = Y > max([pts2_display[2][1],pts2_display[3][1]])
            dst[mask] = (0, 0, 0)
    else:
        if last_M is not None:
            dst = cv2.warpPerspective(img, last_M, (window_width, window_height))
            #裏返りを消す
            sign = np.float32([find_turning_minus(*pts1_in_mm[0], *gaze_in_mm),
                                  find_turning_minus(*pts1_in_mm[1], *gaze_in_mm),
                                  find_turning_minus(*pts1_in_mm[2], *gaze_in_mm),
                                  find_turning_minus(*pts1_in_mm[3], *gaze_in_mm)])
            if flag2 == 0:
                height, width = dst.shape[:2]
                X, Y = np.meshgrid(np.arange(width), np.arange(height))
                flag2 += 1
            if sign[0] > 0 and sign[1] > 0:
                mask = Y < min(pts2_display[0][1],pts2_display[1][1])
                dst[mask] = (0, 0, 0)
            if sign[0] > 0 and sign[2] > 0:
                mask = (pts2_display[0][0]-pts2_display[2][0])*(Y-pts2_display[0][1]) < (pts2_display[0][1]-pts2_display[2][1])*(X-pts2_display[0][0])
                dst[mask] = (0, 0, 0)
            if sign[1] > 0 and sign[3] > 0:
                mask = (pts2_display[1][0]-pts2_display[3][0])*(Y-pts2_display[1][1]) > (pts2_display[1][1]-pts2_display[3][1])*(X-pts2_display[1][0])
                dst[mask] = (0, 0, 0)
            if sign[2] > 0 and sign[3] > 0:
                mask = Y > max([pts2_display[2][1],pts2_display[3][1]])
                dst[mask] = (0, 0, 0)
            
        else:
            dst = img.copy()

    cv2.imshow("Gaze Viewer", dst)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    time.sleep(0.002)

cv2.destroyAllWindows()
