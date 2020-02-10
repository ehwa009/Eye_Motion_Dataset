import cv2
import numpy as np
import dlib
import argparse
import glob
import pickle
import os
import sys

from tqdm import tqdm
from landmarking.calibration import Calibration
from landmarking.eye import Eye


def puplis_located(eye_left, eye_right):
    try:
        int(eye_left.pupil.x)
        int(eye_left.pupil.y)
        int(eye_right.pupil.x)
        int(eye_right.pupil.y)
        return True
    except Exception:
        return False


'''
Extract facial landmarks on detected face.
    {
        'roi': [x1, y1, x2, y2],
        'landmark': landmark object,
        'pulpil': [x1, y1, x2, y2],
    }
'''
def get_landmark_ver_2(frame, detector, predictor, calibration):
    coord_data = {}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) < 1:
        return {}
    else:
        face = faces[0]
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        coord_data['roi'] = [x1, y1, x2, y2] # add face roi coordination
        landmark = predictor(gray, face)
        for idx in range(36, 48):
            if landmark.pard(idx).x < 100 or landmark.part(idx).y < 100:
                return {}
        coord_data['landmark'] = landmark # add landmark object
        # pulpil detection
        eye_left = Eye(gray, landmark, 0, calibration)
        eye_right = Eye(gray, landmark, 1, calibration)
        if puplis_located(eye_left, eye_right):
            eye_left_x, eye_left_y = eye_left.origin[0] + eye_left.pupil.x, eye_left.origin[1] + eye_left.pupil.y
            eye_right_x, eye_right_y = eye_right.origin[0] + eye_right.pupil.x, eye_right.origin[1] + eye_right.pupil.y
            # add pupil location
            coord_data['pulpil'] = [int(eye_left_x), int(eye_left_y), int(eye_right_x), int(eye_right_y)]
        else: # add default 
            coord_data['pulpil'] = [0, 0, 0, 0]

    return coord_data

'''
only draw face roi, eyebrow, eye region, and pupil
'''
def draw_landmark(frame, coord_data):
    roi_dot = coord_data['roi']
    landmark_dot = coord_data['landmark']
    pulpil_dot = coord_data['pulpil']
    left_eye_region = np.array(list(zip([landmark_dot.part(i).x for i in range(36, 42)],
                                        [landmark_dot.part(i).y for i in range(36, 42)])), np.int32)
    right_eye_region = np.array(list(zip([landmark_dot.part(i).x for i in range(42, 48)],
                                        [landmark_dot.part(i).y for i in range(42, 48)])), np.int32)
    left_eyebrow = list(zip([landmark_dot.part(i).x for i in range(17, 21)],
                            [landmark_dot.part(i).y for i in range(17, 21)]))
    right_eyebrow = list(zip([landmark_dot.part(i).x for i in range(22, 27)],
                            [landmark_dot.part(i).y for i in range(22, 27)]))

    # draw roi
    cv2.rectangle(frame, (roi_dot[0], roi_dot[1]), (roi_dot[2], roi_dot[3]))
    # draw eye
    cv2.polylines(frame, [left_eye_region], True, (255, 255 , 255), 1)
    cv2.polylines(frame, [right_eye_region], True, (255, 255, 255), 1)
    cv2.circle(frame, (pulpil_dot[0], pulpil_dot[1]), 3, (255, 255, 255), -1)
    cv2.circle(frame, (pulpil_dot[2], pulpil_dot[3]), 3, (255, 255, 255), -1)
    # draw eyebrow
    for index, item in enumerate(right_eyebrow):
            if index == len(right_eyebrow) - 1:
                break
            cv2.line(frame, item, right_eyebrow[index + 1], (255, 255, 255), 1)

    for index, item in enumerate(left_eyebrow):
        if index == len(left_eyebrow) - 1:
            break
        cv2.line(frame, item, left_eyebrow[index + 1], (255, 255, 255), 1)

'''
option2:
    [...]
    0 ~ 3: pulpil left and right x y coordination
    4 ~ 15: left eye region x y coordination
    16 ~ 27: right eye region x y coordination
    28 ~ 37: right eyebrow
    38 ~ 47: left eyebrow 
    48 ~ 49: center point
'''
def get_landmark(frame, detector, predictor, calibration, opt):
    keypoints = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    ############################### Extract landmarks ###############################
    if len(faces) < 1: # no face detected
        return [0] * 50
    else:
        face = faces[0]
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        landmarks = predictor(gray, face)
        # To prevent detecting only single side of face.
        # It causes some bug to pause when calcuate the coordination of pulpil
        for idx in range(36, 48): 
            if landmarks.part(idx).x < 100 or landmarks.part(idx).y < 100:
                return [0] * 50
        
        # get cooredination of both eye region 
        left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                    (landmarks.part(37).x, landmarks.part(37).y),
                                    (landmarks.part(38).x, landmarks.part(38).y),
                                    (landmarks.part (39).x, landmarks.part(39).y),
                                    (landmarks.part(40).x, landmarks.part(40).y),
                                    (landmarks.part(41).x, landmarks.part(41).y)], np.int32)
        right_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                    (landmarks.part(43).x, landmarks.part(43).y),
                                    (landmarks.part(44).x, landmarks.part(44).y),
                                    (landmarks.part(45).x, landmarks.part(45).y),
                                    (landmarks.part(46).x, landmarks.part(46).y),
                                    (landmarks.part(47).x, landmarks.part(47).y)], np.int32)

        eye_left = Eye(gray, landmarks, 0, calibration)
        eye_right = Eye(gray, landmarks, 1, calibration)
        if puplis_located(eye_left, eye_right):
            eye_left_x, eye_left_y = eye_left.origin[0] + eye_left.pupil.x, eye_left.origin[1] + eye_left.pupil.y
            eye_right_x, eye_right_y = eye_right.origin[0] + eye_right.pupil.x, eye_right.origin[1] + eye_right.pupil.y
            # add pupil location
            keypoints += [int(eye_left_x), int(eye_left_y), int(eye_right_x), int(eye_right_y)]
        else: # add default 
            keypoints += [0, 0, 0, 0]

        # add eye regions
        for er in range(36, 48):
            keypoints += [int(landmarks.part(er).x), int(landmarks.part(er).y)]
        # add eye brow:
        for eb in range(17, 27):
            keypoints += [int(landmarks.part(eb).x), int(landmarks.part(eb).y)]
        # add a center point
        keypoints += [int(landmarks.part(27).x), int(landmarks.part(27).y)]

        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])
        
        ############################### Draw Components ###############################
        # draw face roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # draw eye
        cv2.polylines(frame, [left_eye_region], True, (255, 255 , 255), 1)
        cv2.polylines(frame, [right_eye_region], True, (255, 255, 255), 1)
        
        # draw pupils
        if puplis_located(eye_left, eye_right):
            cv2.circle(frame, (eye_left_x, eye_left_y), 3, (255, 255, 255), -1)
            cv2.circle(frame, (eye_right_x, eye_right_y), 3, (255, 255, 255), -1)

        # draw eye brow
        for n in range(17, 27):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

        cv2.circle(frame, (landmarks.part(27).x, landmarks.part(27).y), 2, (255, 0, 0), -1)

    return keypoints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-vid_path', default='./videos')
    parser.add_argument('-facial_keypoints', default='./facial_keypoints')
    parser.add_argument('-model_path', default='./model/shape_predictor_68_face_landmarks.dat')
    
    parser.add_argument('-width', type=int, default=960)
    parser.add_argument('-height', type=int, default=540)
    parser.add_argument('-frame_threshold', type=int, default=500)
    parser.add_argument('-vid_idx_from', type=int, default=0)
    parser.add_argument('-use_ver_2', type=bool, default=False)
    
    opt = parser.parse_args()
    calibration = Calibration()
    if not(os.path.exists(opt.facial_keypoints)):
        os.mkdir(opt.facial_keypoints)
    videos = glob.glob(opt.vid_path + '/*.mp4')
    print('[INFO] Total number of videos: {}'.format(str(len(videos))))
    
    for i, fp in tqdm(enumerate(sorted(videos, key=os.path.getmtime))):
        if opt.vid_idx_from >= i: # extract landmarks from input vid idx
            continue
        sys.stdout.write('\t{}/{}'.format(i+1, str(len(videos))))
        vid_name = os.path.split(fp)[1][-15:-4]
        print('\n\tCurrent Video: {}'.format(vid_name))
        if not(os.path.exists('{}/{}.pickle'.format(opt.facial_keypoints, vid_name))):
            cap = cv2.VideoCapture(fp)
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(opt.model_path)
            all_keypoints = []
            num_frame = 0
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret:
                    # find and get facial keypoints on a single frame
                    frame = cv2.resize(frame, (opt.width, opt.height), interpolation=cv2.INTER_AREA)
                    if opt.use_ver_2:
                        coord_data = get_landmark_ver_2(frame, detector, predictor, calibration)
                        draw_landmark(frame, coord_data) # draw
                        if (num_frame > opt.frame_threshold) & (coord_data['roi'] == [[0] * 4] * opt.frame_threshold):
                            print('[INFO] There is no face detected. Delete the video and go to next video.')
                            # delete uneccessary videos and subtitles
                            os.remove('{}/{}.mp4'.format(opt.vid_path, vid_name))
                            os.remove('{}/{}.vtt'.format(opt.vid_path, vid_name))
                            break
                    else:
                        coord_data = get_landmark(frame, detector, predictor, calibration, opt)
                        if (num_frame > opt.frame_threshold) & (all_keypoints == [[0] * 50] * opt.frame_threshold):
                            print('[INFO] There is no face detected. Delete the video and go to next video.')
                            # delete uneccessary videos and subtitles
                            os.remove('{}/{}.mp4'.format(opt.vid_path, vid_name))
                            os.remove('{}/{}.vtt'.format(opt.vid_path, vid_name))
                            break
                    num_frame += 1
                    all_keypoints.append(coord_data)
                    cv2.imshow('frame', frame) # Display
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
            cap.release()
            cv2.destroyAllWindows()

            if len(all_keypoints) > 0:
                with open('{}/{}.pickle'.format(opt.facial_keypoints, vid_name), 'wb') as f:
                    pickle.dump(all_keypoints, f)
                    print('[INFO] landmarks saved at {}.'.format(f))
                    # exit(-1) # test purpose 

            
if __name__ == '__main__':
    main()