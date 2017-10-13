import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib inline

images = glob.glob('camera_cal/*.jpg')

objpoints = [];
imgpoints = [];

rows = 6;
cols = 9;

objp = np.zeros((rows*cols,3),np.float32)
objp[:,:2] = np.mgrid[0:rows,0:cols].T.reshape(-1,2)

for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (cols,rows), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (cols,rows), corners, ret)
        write_name = 'cal_out/corners_found'+str(idx)+'.jpg'
        cv2.imwrite(write_name, img)
        #cv2.imshow('img', img)
        #cv2.waitKey(500)

print("Done")

cv2.destroyAllWindows()
