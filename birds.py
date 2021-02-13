import cv2
# from matplotlib import pyplot as plt
import numpy as np
img = cv2.imread("Bird's eye view/t1.png")

# IMAGE_H = 223
# IMAGE_W = 1280

# src = np.float32([[0, IMAGE_H], [1207, IMAGE_H], [0, 0], [IMAGE_W, 0]])
# dst = np.float32([[569, IMAGE_H], [711, IMAGE_H], [0, 0], [IMAGE_W, 0]])
# M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix

# img_ = img[450:(450+IMAGE_H), 0:IMAGE_W] # Apply np slicing for ROI crop
# warped_img = cv2.warpPerspective(img_, M, (IMAGE_W, IMAGE_H)) # Image warping

# while True:
#     cv2.imshow('t1-case', warped_img)
#     if cv2.waitKey(5) & 0xFF == 27: #exit only after 5 miliseconds and on esc key being pressed
#         break

cv2.circle(img, (170,156), 5, (0,0,255), -1)
cv2.circle(img, (1,211), 5, (0,0,255), -1)
cv2.circle(img, (268,152), 5, (0,0,255), -1)
cv2.circle(img, (426,205), 5, (0,0,255), -1)

while True:
    cv2.imshow('t1-case', img)
    if cv2.waitKey(5) & 0xFF == 27: #exit only after 5 miliseconds and on esc key being pressed
        break
while True:
    # pts1 = np.float32([[82, 181], [82, 231], [335, 181], [335, 231]]) 
    pts1 = np.float32([[170,156], [1, 211], [268, 152], [426, 205]]) 
    pts2 = np.float32([[0, 0], [0, 600], [500, 0], [500, 600]]) 
    #                   TL        BL       TR        BR
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2) 
    result = cv2.warpPerspective(img, matrix, (500,600))
    cv2.imshow('t1-sol', result)
    if cv2.waitKey(5) & 0xFF == 27: #exit only after 5 miliseconds and on esc key being pressed
        break

img_fix = cv2.cvtColor(img,
                        cv2.COLOR_BGR2RGB)
hsv_img = cv2.cvtColor(img_fix,
                        cv2.COLOR_RGB2HSV)

light_orange = (0,0,0)
dark_orange = (200, 100, 50)

# # light_orange = (0,0,0)
# # dark_orange = (222,222,222)

mask = cv2.inRange(hsv_img, light_orange, dark_orange)

result = cv2.bitwise_and(img_fix, img_fix, mask=mask)

plt.imshow(mask, cmap="gray")
# while True:
#     cv2.imshow('t1-sol++', result)
#     if cv2.waitKey(5) & 0xFF == 27: #exit only after 5 miliseconds and on esc key being pressed
#         break


################## OPENCV INRANGE FUNCTION ###################
