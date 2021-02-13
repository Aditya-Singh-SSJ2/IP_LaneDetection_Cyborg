import cv2
import numpy as np

vidcap = cv2.VideoCapture("Bird's eye view/imgproc_samplevid.mp4")
success,image = vidcap.read()
count = 0
print("I am in success")
while success:
  success,image = vidcap.read()
  resize = cv2.resize(image, (640, 480)) 
#   cv2.imwrite("%03d.jpg" % count, resize) 
  cv2.imshow("Result", resize)    
  if cv2.waitKey(10) == 27:                     
      break
  count += 1