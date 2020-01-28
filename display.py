import cv2
import numpy as np
import argparse
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-landmarks_path', default='./facial_keypoints/Xo9J_G1cTsk.pickle')
    parser.add_argument('-x_lim', default=960)
    parser.add_argument('-y_lim', default=800)
    opt = parser.parse_args()

    with open(opt.landmarks_path, 'rb') as f:
        landmarks = pickle.load(f)

    for landmark in landmarks:
        frame = np.zeros((opt.x_lim, opt.y_lim, 3), np.uint8)
        
        left_eye_region = np.array(list(zip(landmark[4:16:2], landmark[5:16:2])), np.int32)
        right_eye_region = np.array(list(zip(landmark[16:28:2], landmark[17:28:2])), np.int32)
        
        right_pupil = np.array(landmark[0:2], np.int32)
        left_pupil = np.array(landmark[2:4], np.int32)
        
        right_eyebrow = list(zip(landmark[28:38:2], landmark[29:38:2]))
        left_eyebrow = list(zip(landmark[38:48:2], landmark[39:48:2]))

        center_dot = (landmark[48], landmark[29])

        cv2.polylines(frame, [left_eye_region], True, (255, 255, 255), 1)
        cv2.polylines(frame, [right_eye_region], True, (255, 255, 255), 1)
        
        cv2.circle(frame, (left_pupil[0], left_pupil[1]), 3, (255, 255, 255), -1)
        cv2.circle(frame, (right_pupil[0], right_pupil[1]), 3, (255, 255, 255), -1)

        for index, item in enumerate(right_eyebrow):
            if index == len(right_eyebrow) - 1:
                break
            cv2.line(frame, item, right_eyebrow[index + 1], (255, 255, 255), 1)

        for index, item in enumerate(left_eyebrow):
            if index == len(left_eyebrow) - 1:
                break
            cv2.line(frame, item, left_eyebrow[index + 1], (255, 255, 255), 1)

        cv2.circle(frame, center_dot, 2, (255, 0, 0), -1)

        cv2.imshow('display', frame)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
