import cv2
import numpy as np
import argparse
import pickle
import glob
import random

class Display:

    def __init__(self, x_lim, y_lim, sp=30):
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.sp = sp

        # font settig
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.bottomLeftCornerOfText = (int(self.x_lim/54), int(self.y_lim/2))
        self.topLeftConnerOfText = (int(self.x_lim/54), int(self.y_lim/19))
        self.fontScale = 0.4
        self.fontColor = (255,255,255)
        self.lineType = 1
            
    def draw_frame(self, landmark, is_center, text=None, title=None):
        frame = np.zeros((self.x_lim, self.y_lim, 3), np.uint8)
        
        left_eye_region = np.array(list(zip(landmark[4:16:2], landmark[5:16:2])), np.int32)
        right_eye_region = np.array(list(zip(landmark[16:28:2], landmark[17:28:2])), np.int32)
        # draw on frame
        cv2.polylines(frame, [left_eye_region], True, (255, 255, 255), 1)
        cv2.polylines(frame, [right_eye_region], True, (255, 255, 255), 1)

        right_pupil = np.array(landmark[0:2], np.int32)
        left_pupil = np.array(landmark[2:4], np.int32)
        cv2.circle(frame, (left_pupil[0], left_pupil[1]), 3, (255, 255, 255), -1)
        cv2.circle(frame, (right_pupil[0], right_pupil[1]), 3, (255, 255, 255), -1)

        right_eyebrow = list(zip(landmark[28:38:2], landmark[29:38:2]))
        for index, item in enumerate(right_eyebrow):
            if index == len(right_eyebrow) - 1:
                break
            cv2.line(frame, item, right_eyebrow[index + 1], (255, 255, 255), 1)
        
        left_eyebrow = list(zip(landmark[38:48:2], landmark[39:48:2]))
        for index, item in enumerate(left_eyebrow):
            if index == len(left_eyebrow) - 1:
                break
            cv2.line(frame, item, left_eyebrow[index + 1], (255, 255, 255), 1)

        if is_center:
            center_dot = (landmark[48], landmark[29])
            cv2.circle(frame, center_dot, 2, (255, 0, 0), -1)

        # put text
        if text:
            cv2.putText(frame, text, 
                        self.bottomLeftCornerOfText, 
                        self.font, 
                        self.fontScale,
                        self.fontColor,
                        self.lineType)

        # put current video text
        if title:
            cv2.putText(frame, 'Current_vid: {}'.format(title), 
                        self.topLeftConnerOfText, 
                        self.font, 
                        self.fontScale,
                        self.fontColor,
                        self.lineType)

        return frame

    def display_dataset(self, dataset_path, is_center=True):
        with open(dataset_path, 'rb') as f:
            eye_dataset = pickle.load(f)
        for ed in eye_dataset:
            print('[INFO] Current video: {}'.format(ed['vid']))
            for ci in ed['clip_info']:
                for sent, landmarks in zip(ci['sent'], ci['landmarks']):
                    for landmark in landmarks:
                        frame = self.draw_frame(landmark, is_center, sent[2], ed['vid'])
                        cv2.imshow('display', frame)
                        if cv2.waitKey(self.sp) & 0xFF == ord('q'):
                            exit(-1)

    def display_processed_dataset(self, dataset_path, is_center=False):
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        eye_dataset = dataset['eye_dataset']
        estimator = dataset['estimator']

        # for ed in eye_dataset:
        #     print('[INFO] Current video: {}'.format(ed['vid']))
        #     for ci in ed['clip_info']:
        #         for sent, landmarks in zip(ci['sent'], ci['landmarks']):
        #             for landmark in landmarks:
        #                 transformed = estimator.inverse_transform(np.array([landmark]))
        #                 transformed = [int(trans) for trans in transformed.tolist()[0]]
        #                 frame = self.draw_frame(transformed, is_center, sent[2], ed['vid'])
        #                 cv2.imshow('display', frame)
        #                 if cv2.waitKey(self.sp) & 0xFF == ord('q'):
        #                     break

        for i in range(len(eye_dataset)):
            video = random.choice(eye_dataset)
            for ci in video['clip_info']:
                for sent, landmarks in zip(ci['sent'], ci['landmarks']):
                    for landmark in landmarks:
                        transformed = estimator.inverse_transform(np.array([landmark]))
                        transformed = [int(trans) for trans in transformed.tolist()[0]]
                        frame = self.draw_frame(transformed, is_center, sent[2], video['vid'])
                        cv2.imshow('display', frame)
                        if cv2.waitKey(self.sp) & 0xFF == ord('q'):
                            break



if __name__ == '__main__':
    # d = Display(540, 960, sp=50) # 960 x 540
    d = Display(180, 320, sp=30) # 320 x 180
    
    # facial_data_list = glob.glob('./facial_keypoints/*.pickle')
    # facial_data = random.choice(facial_data_list)
    # with open(facial_data, 'rb') as f:
    #     landmarks = pickle.load(f)
    # d.run_display(landmarks)

    # dataset = './dataset/processed_eye_motion_dataset.pickle'
    dataset = './dataset/processed_eye_motion_dataset_pca_7.pickle'
    # d.display_dataset(dataset, is_center=False)
    d.display_processed_dataset(dataset)

    
