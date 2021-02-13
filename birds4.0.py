import cv2
import numpy as np

cap = cv2.VideoCapture("Bird's eye view/ip.mp4",0)
success, frame = cap.read()

def nothing(x):
    pass

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)





while True:
    success, frame = cap.read()

    frame = cv2.resize(frame, (640, 480)) 

    tl = (225,387)
    bl = (70,472)
    tr = (400,380)
    br = (538,472)

    frame2 = frame.copy()
    cv2.circle(frame2, tl, 5, (0,255,0), -1)
    cv2.circle(frame2, bl, 5, (0,255,0), -1)
    cv2.circle(frame2, tr, 5, (0,255,0), -1)
    cv2.circle(frame2, br, 5, (0,255,0), -1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame = cv2.resize(frame, (640, 480)) 

    pts1 = np.float32([tl, bl, tr, br]) 
    pts2 = np.float32([[0, 0], [0, 600], [500, 0], [500, 600]]) 
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2) 
    result2 = cv2.warpPerspective(hsv, matrix, (500,600))
    
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
    mask = cv2.inRange(result2, lower, upper)
    mask = cv2.resize(mask, (640, 480)) 
    result = cv2.bitwise_and(frame, frame, mask = mask)

    

    # cv2.imshow("Birds'Eye_HSV", result2)
    
    # cv2.imshow("Result", result)
    
    # if cv2.waitKey(5) & 0xFF == 27: #exit only after 5 miliseconds and on esc key being pressed
    #     break
    # gray = cv2.cvtColor(result2, cv2.COLOR_BGR2GRAY)
    # leftx = 0
    # rightx = 0

    # if leftx==0 and rightx==0:
    histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint


    y = 472
    lx = []
    rx = []

    msk = mask.copy()

    # i = 0
    while y>0:

        # Left threshhold
        img = mask[ y-40 : y, left_base-50 : left_base+50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                lx.append(left_base-50 + cX)
                left_base = left_base-50 + cX

        # print(i, left_base)
        # i += 1

        # Right threshhold
        img = mask[ y-40 : y, right_base-50 : right_base+50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                rx.append(right_base-50 + cX)
                right_base = right_base-50 + cX

        cv2.rectangle(msk,  (left_base-50, y),   (left_base+50, y-40), (255,0,0),    2)
        cv2.rectangle(msk,  (right_base-50, y),   (right_base+50, y-40), (255,0,0),    2)
        
        y-=40

    
    cv2.imshow("Frame", frame2)
    cv2.imshow("Mask", msk)

    # # Draw Rectanglar windos here



    # # Polyfit all the points from windows
    # left_fit = np.polyfit(lefty, leftx, 2)
    # right_fit = np.polyfit(righty, rightx, 2)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()