# Program for detecting yellow ball on a green mat and displaying its XY coordinates on the video
import cv2 as cv
import numpy as np

#Done on 80 cm ball video shared
cap = cv.VideoCapture(
    'stereo_vision/new_camera_data/stereo_camera_test_230208/80cm/stereo_80cm_ball_230208_0218pm.avi')

# if result video need to save to disk change write_video=1 else write_video=0
write_video = 0

# initializing video writer
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))//2
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))
left_v = cv.VideoWriter('left.avi', cv.VideoWriter_fourcc(
    *'XVID'), fps, (width, height))

# function for color mask


def mask(img, color):
    # convert the image to HSV

    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    kernal = np.ones((5, 5), "uint8")

    if color == 'green':
        # define range of green color in HSV
        lower_green = np.array([45, 100, 50])
        upper_green = np.array([75, 255, 255])
        # Threshold the HSV image to get only green colors
        mask = cv.inRange(hsv_img, lower_green, upper_green)
    elif color == 'yellow':
        yellow_lower = np.array([23, 41, 133])
        yellow_upper = np.array([40, 150, 255])
        mask = cv.inRange(hsv_img, yellow_lower, yellow_upper)

    mask = cv.dilate(mask, kernal)
    return mask


def ball_detect(left_frame, right_frame):
    left_ball = mask(left_frame, 'yellow')
    right_ball = mask(right_frame, 'yellow')

    def draw_rect(frame):
        x, y, w, h = 0, 0, 0, 0
        contours, hierarchy = cv.findContours(
            frame,  cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contour = max(contours, key=cv.contourArea)
            area = cv.contourArea(contour)

            # filtering object by its area
            if area > 500 and area < 10000:
                x, y, w, h = cv.boundingRect(contour)
        return (x, y, w, h)

    l_det = draw_rect(left_ball)
    r_det = draw_rect(right_ball)

    return l_det, r_det


def _map(x, in_min, in_max, out_min, out_max):
    a = int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)
    return a

try:
    while True:
        # reading video source
        ret, frame = cap.read()
        if not ret:
            print("no video")
            break

        # slicing stereo video to left and right frames
        left_frame = frame[0:frame.shape[0], 0:(frame.shape[1]//2)]
        right_frame = frame[0:frame.shape[0],
                            (frame.shape[1]//2):frame.shape[1]]

        # send left and right frames to ball detection function
        left_detection, right_detection = ball_detect(left_frame, right_frame)

        # using detected rectangles
        (lx, ly, lw, lh) = left_detection
        (x, y, w, h) = right_detection

        # getting rectangle center
        lcenter = (lx+lw//2, ly+lh//2)
        rcenter = (x+w//2, y+h//2)

        # Preset origin point
        r_org = (725, 67)  # origin point in right frame
        l_org = (707, 102)  # origin point in left frame

        # drawing reference axis on the image
        cv.line(left_frame, l_org, (l_org[0]+50, l_org[1]), (0, 0, 255), 2)
        cv.line(left_frame, l_org, (l_org[0], l_org[1]+50), (255, 0, 0), 2)

        Lx, Ly, Lz = 0, 0, 0
        if lcenter != (0, 0):

            # mapping orgin point Y to the image's height to 0 to 100cm
            Ly = _map(lcenter[1], l_org[1], 720, 0, 100)

            # mapping orgin point X to the image's width to 0 to 100cm
            Lx = _map(lcenter[0], l_org[0], 1280, 0, 100)

            # display bounding box over detected ball
            cv.rectangle(left_frame, (lx, ly), (lx+lw, ly+lh), (255, 0, 0), 2)

            # Display X and Y in frame
            cv.putText(left_frame, "X: {} cm".format(str(round(Lx, 1))), (lx+w+15, lcenter[1]-30), cv.FONT_HERSHEY_SIMPLEX,
                       fontScale=1, color=[0, 0, 255], thickness=1, lineType=cv.LINE_AA)
            cv.putText(left_frame, "Y: {} cm".format(str(round(Ly, 1))), (lx+w+15, lcenter[1]), cv.FONT_HERSHEY_SIMPLEX,
                       fontScale=1, color=[225, 0, 0], thickness=1, lineType=cv.LINE_AA)

        print("X,Y: ", Lx, Ly)
        cv.imshow("L", left_frame)

        # result video write to disk
        if write_video:
            left_v.write(left_frame)

        k = cv.waitKey(1)
        if k == ord('q'):
            break

finally:
    if write_video:
        left_v.release()
    cap.release()
    cv.destroyAllWindows()
