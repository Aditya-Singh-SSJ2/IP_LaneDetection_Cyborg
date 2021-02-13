import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def nothing(x):
    pass

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("U - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("U - V", "Trackbars", 0, 179, nothing)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame = cv2.resize(frame, (640, 480)) 
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    
    # lower = [0,0,0]
    # upper = [179,179,179]

    # cv2.circle(frame, (100,100), 5, (0,0,255), -1)
    # pts1 = np.float32([[],[],[],[]])
    lower = np.array([l_h,l_s,l_v])
    upper = np.array([u_h,u_s,u_v])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.resize(mask, (640, 480)) 
    result = cv2.bitwise_and(frame, frame, mask = mask)


    pts1 = np.float32([[170,156], [1, 211], [268, 152], [426, 205]]) 
    pts2 = np.float32([[0, 0], [0, 600], [500, 0], [500, 600]]) 
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2) 
    result2 = cv2.warpPerspective(frame, matrix, (500,600))
    
    cv2.imshow('t1-sol', result2)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)
    # if cv2.waitKey(5) & 0xFF == 27: #exit only after 5 miliseconds and on esc key being pressed
    #     break
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
