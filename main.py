import cv2
from color_hsv import bgr_to_hsv
import numpy as np

yellow = [6, 147, 194] #BGR format of yellow color
pink = [142, 154, 183]   #BGR format of pink color
black = [60, 55, 52]      #BGR format of black color
blue = [150, 77, 26]     #BGR format of blue color
green = [87, 116, 17]   #BGR format of green color
brown = [70, 75, 82] #BGR format of brown color

color = [
    "black",
    "pink",
    "blue",
    "brown",
    "green",
    "yellow"
]

lower_limit_ylw, upper_limit_ylw =bgr_to_hsv(yellow)
lower_limit_pnk, upper_limit_pnk =bgr_to_hsv(pink)
lower_limit_blk, upper_limit_blk =bgr_to_hsv(black)
lower_limit_blu, upper_limit_blu =bgr_to_hsv(blue)
lower_limit_grn, upper_limit_grn =bgr_to_hsv(green)
lower_limit_brn, upper_limit_brn =bgr_to_hsv(brown)

ranges = [
    (lower_limit_blk, upper_limit_blk),
    (lower_limit_pnk, upper_limit_pnk),
    (lower_limit_blu, upper_limit_blu),
    (lower_limit_brn, upper_limit_brn),
    (lower_limit_grn, upper_limit_grn),
    (lower_limit_ylw, upper_limit_ylw)
]

img = cv2.imread('1_1.png')
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
img_gray = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
v = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
low = 0.5 * v
high = 1.5 * v
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))

canny_img = cv2.Canny(img_gray, low, high)
canny_img = cv2.morphologyEx(canny_img, cv2.MORPH_CLOSE, kernel, 4)
contours_canny = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
results = np.empty((0, 3))

for cnt in contours_canny:
    j = 0
    mask = []
    if cv2.contourArea(cnt) > 150:
        x, y, w, h = cv2.boundingRect(cnt)
        w_center = x + w // 2
        h_center = y + h // 2

        hsv_color = img[y:y+h, x:x+w]
        hsv_color = cv2.cvtColor(hsv_color, cv2.COLOR_BGR2HSV)

        for i, (l_lim, u_lim) in enumerate(ranges):
            mask_color = cv2.inRange(hsv_color, l_lim, u_lim)
            mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, kernel, 4)
            mask.append(mask_color)
        areas = [np.count_nonzero(m) for m in mask]
        max_index = np.argmax(areas)
        cv2.putText(img, color[max_index], (w_center - 20, h_center - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # print(f"Color {color[max_index]}: Center=({w_center}, {h_center}), Width={w}, Height={h}")
        new_row = np.array([[w_center, h_center, max_index]])
        results = np.vstack([results, new_row])
        cv2.circle(img, (w_center, h_center), 3, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)

print(results.astype(int))
cv2.imshow('Image', img) 
cv2.waitKey(0)
cv2.destroyAllWindows()
