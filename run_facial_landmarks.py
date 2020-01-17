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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-vid_path', default='./videos')
    parser.add_argument('-facial_keypoints', default='./facial_keypoints')
    parser.add_argument('-model_path', default='./model/shape_predictor_68_face_landmarks.dat')
    opt = parser.parse_args()

    calibration = Calibration()

    if not(os.path.exists(opt.facial_keypoints)):
        os.mkdir(opt.facial_keypoints)

    videos = glob.glob(opt.vid_path + '/*.mp4')
    print('[INFO] Total number of videos: {}'.format(str(len(videos))))
    
    for i, fp in tqdm(enumerate(sorted(videos, key=os.path.getmtime))):
        sys.stdout.write('{}/{}'.format(i+1, str(len(videos))))
        vid_name = os.path.split(fp)[1][-15:-4]
        print('[INFO] Current Video: {}'.format(vid_name))

        cap = cv2.VideoCapture(fp)

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(opt.model_path)
        
        all_keypoints = []
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                # find and get facial keypoints on a single frame
                facial_keypoints = get_landmark(frame, detector, predictor, calibration)
                if facial_keypoints != []:
                    all_keypoints.append(facial_keypoints)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        
        cap.release()
        cv2.destroyAllWindows()

        with open('{}/{}.pickle'.format(opt.facial_keypoints, vid_name), 'wb') as f:
            pickle.dump(all_keypoints, f)


def puplis_located(eye_left, eye_right):
    try:
        int(eye_left.pupil.x)
        int(eye_left.pupil.y)
        int(eye_right.pupil.x)
        int(eye_right.pupil.y)
        return True
    except Exception:
        return False


def get_landmark(frame, detector, predictor, calibration):
    keypoints = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    try:
        face = faces[0]
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        
        landmarks = predictor(gray, face)
        eye_left = Eye(gray, landmarks, 0, calibration)
        eye_right = Eye(gray, landmarks, 1, calibration)
        
        # eye_left_x, eye_left_y, eye_right_x, eye_right_y = None, None, None, None
        if puplis_located(eye_left, eye_right):
            eye_left_x, eye_left_y = eye_left.origin[0] + eye_left.pupil.x, eye_left.origin[1] + eye_left.pupil.y
            eye_right_x, eye_right_y = eye_right.origin[0] + eye_right.pupil.x, eye_right.origin[1] + eye_right.pupil.y
            
            # add pupil location
            keypoints += [eye_left_x, eye_left_y, eye_right_x, eye_right_y]

        # gaze detection
        left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                    (landmarks.part(37).x, landmarks.part(37).y),
                                    (landmarks.part(38).x, landmarks.part(38).y),
                                    (landmarks.part(39).x, landmarks.part(39).y),
                                    (landmarks.part(40).x, landmarks.part(40).y),
                                    (landmarks.part(41).x, landmarks.part(41).y)], np.int32)
        
        right_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                    (landmarks.part(43).x, landmarks.part(43).y),
                                    (landmarks.part(44).x, landmarks.part(44).y),
                                    (landmarks.part(45).x, landmarks.part(45).y),
                                    (landmarks.part(46).x, landmarks.part(46).y),
                                    (landmarks.part(47).x, landmarks.part(47).y)], np.int32)
        # add eye regions
        for er in range(36, 48):
            keypoints.append(landmarks.part(er).x)
            keypoints.append(landmarks.part(er).y)

        # add eye brow:
        for eb in range(17, 27):
            keypoints.append(landmarks.part(eb).x)
            keypoints.append(landmarks.part(eb).y)

        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])
        
        ############################### Draw Components ###############################
        # draw face roi
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # draw eye
        cv2.polylines(frame, [left_eye_region], True, (0, 0 , 255), 1)
        cv2.polylines(frame, [right_eye_region], True, (0, 0 , 255), 1)
        
        # draw pupils
        if puplis_located(eye_left, eye_right):
            cv2.circle(frame, (eye_left_x, eye_left_y), 3, (0, 0, 255), -1)
            cv2.circle(frame, (eye_right_x, eye_right_y), 3, (0, 0, 255), -1)

        # draw eye brow
        for n in range(17, 27):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
        
        # display eye region
        # eye = frame[min_y:max_y, min_x:max_x]
        # cv2.imshow('left_eye', eye)
        cv2.imshow('video', frame)

    
    except:
        pass

    return keypoints

            
if __name__ == '__main__':
    main()