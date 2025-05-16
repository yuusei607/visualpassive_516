import time
import sys
import threading
import queue
from datetime import datetime, timedelta
import json
import math
import numpy as np
import argparse

import cv2
# import tobii_research as trexport CPPFLAGS="-I$(brew --prefix tcl-tk)/include"
import pyautogui as pag
# import tkinter as tk÷
import zmq
import msgpack
import cv2


# Pupil Core の Network API に接続
ctx = zmq.Context()
ip = 'localhost'
port = 50020

# REQ ソケットで SUB/PUB ポートを取得
pupil_remote = ctx.socket(zmq.REQ)
pupil_remote.connect(f"tcp://{ip}:{port}")

# サブポート取得
pupil_remote.send_string('SUB_PORT')
sub_port = pupil_remote.recv_string()

# サブスクライバ設定
subscriber = ctx.socket(zmq.SUB)
subscriber.connect(f"tcp://{ip}:{sub_port}")
subscriber.subscribe('gaze.')

# ディスプレイサイズを取得（OpenCVでフルスクリーン表示用）
screen_width = 1440  # 適宜あなたの画面に合わせて変更
screen_height = 900

# フルスクリーンウィンドウの準備
cv2.namedWindow("Gaze Viewer", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Gaze Viewer", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    
    # できるだけ最新のメッセージを取得する
    while True:
        try:
            topic, payload = subscriber.recv_multipart(flags=zmq.NOBLOCK)
            message = msgpack.loads(payload, raw=False)
            norm_pos = message.get("norm_pos")
            if norm_pos:
                latest_norm_pos = norm_pos
        except zmq.Again:
            break


    if norm_pos:
        x_norm, y_norm = norm_pos
        # 画面座標に変換
        x = int(x_norm * screen_width)
        y = int((1 - y_norm) * screen_height)  # Y軸は上が0なので反転

        # 赤い円を描画
        cv2.circle(frame, (x, y), radius=20, color=(0, 0, 255), thickness=-1)

    # 表示
    cv2.imshow("Gaze Viewer", frame)

    # 'q' を押すと終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()