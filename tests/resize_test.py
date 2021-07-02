import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fmethods
import input
import numpy as np
import cv2

vs = input.VideoFile("videos/ignored_amal1998.mp4")

for i in range(1200):
    _ = vs.get_frame()
f = vs.get_frame()
f_dims = np.shape(f)[:2]
asp_rat = float(f_dims[1])/f_dims[0]

# Use imutils.resize to fix aspect ratio
f1 = fmethods.resize(f, height=None, width=500)
f1_dims = np.shape(f)[:2]

# FIXED ASPECT RATIO RESIZE
assert np.isclose(float(f1_dims[1])/f1_dims[0], asp_rat)

# FIXED WIDTH RESIZE
# Larger asp ratio
f_larfw = fmethods.c_resize(f1, (int(500 / (asp_rat * 1.2)), 500))
# Smaller asp ratio
f_sarfw = fmethods.c_resize(f1, (int(500 / (asp_rat * 0.8)), 500))

# FIXED HEIGHT RESIZE
f1_height = np.shape(f1)[0]
# Larger asp ratio
f_larfh = fmethods.c_resize(f1, (f1_height, int(f1_height * asp_rat * 1.2)))
# Smaller asp ratio
f_sarfh = fmethods.c_resize(f1, (f1_height, int(f1_height * asp_rat * 0.8)))

cv2.imshow("Resized", f1)
cv2.imshow("Larger Asp. Ratio, fixed width", f_larfw)
cv2.imshow("Smaller Asp. Ratio, fixed width", f_sarfw)
cv2.imshow("Larger Asp. Ratio, fixed height", f_larfh)
cv2.imshow("Smaller Asp. Ratio, fixed height", f_sarfh)
cv2.waitKey(0)
