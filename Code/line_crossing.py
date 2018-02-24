import cv2
import numpy as np
import math

def slope(p1, p2):
    m = float(0)
    if p2[0]-p1[0] != 0:
        m = float((p2[1]-p1[1])/(p2[0]-p1[0]))
        m = math.ceil(m * 100) / 100
    return m


def detect_line_crossing(line_p1, line_p2, x, y, w,h):
    return slope(line_p1, line_p2)

