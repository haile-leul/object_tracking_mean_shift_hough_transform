import numpy as np
import cv2
from utils import *

roi_defined = False
 
def define_ROI(event, x, y, flags, param):
	global r,c,w,h,roi_defined
	# if the left mouse button was clicked, 
	# record the starting ROI coordinates 
	if event == cv2.EVENT_LBUTTONDOWN:
		r, c = x, y
		roi_defined = False
	# if the left mouse button was released,
	# record the ROI coordinates and dimensions
	elif event == cv2.EVENT_LBUTTONUP:
		r2, c2 = x, y
		h = abs(r2-r)
		w = abs(c2-c)
		r = min(r,r2)
		c = min(c,c2)  
		roi_defined = True

cap = cv2.VideoCapture('VOT-Ball.mp4')

# take first frame of the video
ret,frame = cap.read()
# load the image, clone it, and setup the mouse callback function
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)
 
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("First image", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the ROI is defined, draw it!
	if (roi_defined):
		# draw a green rectangle around the region of interest
		cv2.rectangle(frame, (r,c), (r+h,c+w), (0, 255, 0), 2)
	# else reset the image...
	else:
		frame = clone.copy()
	# if the 'q' key is pressed, break from the loop
	if key == ord("q"):
		break
 
track_window = (r,c,h,w)
# set up the ROI for tracking
roi = frame[c:c+w, r:r+h]

# Setup the termination criteria: either 10 iterations,
# or move by less than 1 pixel
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

cpt = 1

while(1):
    ret ,frame = cap.read()
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        gX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
        gY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)


        mag, angle = cv2.cartToPolar(gX, gY, angleInDegrees=True)
        
        masked_orientation = np.where(mag>50, angle, 0)
        masked_orientation_red = cv2.cvtColor(masked_orientation, cv2.COLOR_GRAY2BGR)
        masked_orientation_red = np.where(masked_orientation_red==(0,0,0), (0,0,255), masked_orientation_red)
        
        mag = cv2.convertScaleAbs(mag)
        angle = cv2.convertScaleAbs(angle)
        masked_orientation_red = cv2.convertScaleAbs(masked_orientation_red)

        # Draw a blue rectangle on the current image
        # frame_tracked = cv2.rectangle(frame, (cstart,rstart), (cend,rend), (255,0,0) ,2)
        cv2.imshow('Sequence',frame)
        cv2.imshow('Gradient Norm',mag)
        cv2.imshow('Gradient Orientation',angle)
        cv2.imshow('Selected Orientations',masked_orientation_red)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('Frame_%04d.png'%cpt,frame)
        cpt += 1
    else:
        break

cv2.destroyAllWindows()
cap.release()