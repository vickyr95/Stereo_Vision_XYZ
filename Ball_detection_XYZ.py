# Program for detecting yellow ball on a green mat and displaying its XY coordinates on the video
import cv2 as cv
from helper.ball_detector import ball_detect
from helper.utils import DLT, get_projection_matrix, read_camera_parameters
import numpy as np

# Done on 80 cm ball video shared
cap = cv.VideoCapture(
    'stereo_vision/new_camera_data/stereo_camera_test_230208/80cm/stereo_80cm_ball_230208_0218pm.avi')

# if result video need to save to disk change write_video=1 else write_video=0
write_video = 1

# if Z value is refence from camera frame camera_to_world=0 else camera_to_world=1
camera_to_world = 0

# initializing video writer
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))//2
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))
left_v = cv.VideoWriter('output_xyz.avi', cv.VideoWriter_fourcc(
    *'XVID'), fps, (width, height))

# Function for getting XYZ coords from detection points


def get_xyz(r_pt, l_pt):

    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)

    # RT matrix for C1 is identity.
    # cmtx, dist = read_camera_parameters(0)
    # RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    # P0 = cmtx @ RT1 #projection matrix for C1

    if l_pt[0] == 0 or r_pt[0] == 0:
        point_3d = [0, 0, 0]
    else:
        point_3d = DLT(P0, P1, r_pt, l_pt)  # calculate 3d position of keypoint

    return point_3d

# function for mapping values


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

        img_width = left_frame.shape[1]
        img_height = left_frame.shape[0]

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

        # getting origin's XYZ
        org_xyz = get_xyz(r_org, l_org)
        [oX, oY, oZ] = org_xyz

        # getting ball's XYZ
        ball_xyz = get_xyz(lcenter, rcenter)
        [bX, bY, bZ] = ball_xyz

        Lx, Ly, Lz = 0, 0, 0

        # if ball detected
        if any(ball_xyz) != 0:

            # mapping orgin point Y to the image's height to 0 to 100cm
            Ly = _map(lcenter[1], l_org[1], img_height, 0, 100)

            # mapping orgin point X to the image's width to 0 to 100cm
            Lx = _map(lcenter[0], l_org[0], img_width, 0, 50)

            # ball reference from origin
            if camera_to_world == 0:
                ball_left_xyz = get_xyz(
                    (lcenter[0]-lw, lcenter[1]), (rcenter[0]-w, rcenter[1]))
                ball_right_xyz = get_xyz(
                    (lcenter[0]+lw, lcenter[1]), (rcenter[0]+w, rcenter[1]))
                Lz = abs(ball_xyz[2]-((ball_left_xyz[2]+ball_right_xyz[2])//2))
                # Lz=bZ-oZ

            # ball reference from camera
            else:
                Lz = bZ

            # display bounding box over detected ball
            cv.rectangle(left_frame, (lx, ly), (lx+lw, ly+lh), (255, 0, 0), 2)

            # Display XYZ in frame
            cv.putText(left_frame, "X: {} cm".format(str(round(Lx, 1))), (lx+w+15, lcenter[1]-30), cv.FONT_HERSHEY_SIMPLEX,
                       fontScale=1, color=[0, 0, 255], thickness=1, lineType=cv.LINE_AA)
            cv.putText(left_frame, "Y: {} cm".format(str(round(Ly, 1))), (lx+w+15, lcenter[1]), cv.FONT_HERSHEY_SIMPLEX,
                       fontScale=1, color=[225, 0, 0], thickness=1, lineType=cv.LINE_AA)
            cv.putText(left_frame, "Z: {} cm".format(str(round(Lz, 1))), (lx+w+15, lcenter[1]+30), cv.FONT_HERSHEY_SIMPLEX,
                       fontScale=1, color=[255, 255, 0], thickness=1, lineType=cv.LINE_AA)

        print("X,Y,Z: ", Lx, Ly, Lz)
        cv.imshow("L", left_frame)
        # cv.imshow("R",right_frame)

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
