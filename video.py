import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO

yolo = YOLO()
#-------------------------------------#
#   调用摄像头
capture=cv2.VideoCapture("drive.mp4")
#-------------------------------------#
#capture=cv2.VideoCapture(0)
fps = 0.0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = int(capture.get(cv2.CAP_PROP_FPS))
video = cv2.VideoWriter('result.mp4', fourcc, fps, (1920, 1080))

while(capture.isOpened()):
    t1 = time.time()
    # 读取某一帧
    ret,frame=capture.read()
    if not ret:
        break
    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))
    # 进行检测
    frame = np.array(yolo.detect_image(frame))
    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    fps  = ( fps + (1./(time.time()-t1)) ) / 2
    print("fps= %.2f"%(fps))
    frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # cv2.imshow("video",frame)

    
    video.write(frame)

    #if cv2.waitKey(int(fps)) == 27:
    #    break
    c= cv2.waitKey(1) & 0xff
    if c==27:
      break
capture.release()
video.release()
