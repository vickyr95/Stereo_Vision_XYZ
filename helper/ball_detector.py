import cv2
import numpy as np

# function for color mask
def mask(img, color):
    # convert the image to HSV

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    kernal = np.ones((5, 5), "uint8")

    if color == 'green':
        # define range of green color in HSV
        lower_green = np.array([45, 100, 50])
        upper_green = np.array([75, 255, 255])
        # Threshold the HSV image to get only green colors
        mask = cv2.inRange(hsv_img, lower_green, upper_green)
    elif color == 'yellow':
        yellow_lower = np.array([23, 41, 133])
        yellow_upper = np.array([40, 150, 255])
        mask = cv2.inRange(hsv_img, yellow_lower, yellow_upper)

    mask = cv2.dilate(mask, kernal)
    return mask


def ball_detect(left_frame,right_frame):
    left_ball=mask(left_frame,'yellow')
    right_ball=mask(right_frame,'yellow')
    

    def draw_rect(frame):
        x,y,w,h=0,0,0,0
        contours, hierarchy = cv2.findContours(frame,  cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)>0:
            contour= max(contours,key=cv2.contourArea)
            area = cv2.contourArea(contour)
            if area> 500 and area < 10000: 
                x,y,w,h = cv2.boundingRect(contour)	        

        return (x,y,w,h)
    
    l_det=draw_rect(left_ball)             
    r_det=draw_rect(right_ball)
    
    return l_det,r_det
