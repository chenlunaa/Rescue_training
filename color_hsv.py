import cv2
import numpy as np

def bgr_to_hsv(color):
    c = np.uint8([[color]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    
    h = int(hsvC[0][0][0])
    s = int(hsvC[0][0][1])
    v = int(hsvC[0][0][2])

    lower_s = max(s - 50, 0)
    upper_s = min(s + 50, 255)
    lower_v = max(v - 50, 0)
    upper_v = min(v + 50, 255)
    lower_h = max(h - 10, 0)
    upper_h = min(h + 10, 179)

    lower_limit = np.array([lower_h, lower_s, lower_v], dtype=np.uint8)
    upper_limit = np.array([upper_h, upper_s, upper_v], dtype=np.uint8)

    # lower_limit = np.array(lower_limit, dtype=np.uint8)
    # upper_limit = np.array(upper_limit, dtype=np.uint8)

    return lower_limit, upper_limit 