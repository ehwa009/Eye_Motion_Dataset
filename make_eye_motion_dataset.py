import os
import glob
import argparse
import cv2
import pickle
import numpy as np

from tqdm import tqdm_gui
from data_utils import VideoWrapper, SubtitleWrapper, ClipWrapper


'''
# Eye motion dataset strucuture

{
    'vid': vid_name,
    'clip_info': [
        {
            'sent': clip_sent_list, # [[start_frame, end_frame, sentence], ...]
            'landmarks': clip_landmarks, # [[landmark_list], ...]
            'start_frame': start_frame,
            'end_frame': end_frame,
        },
        ...    
    ]
},
...

'''


def make_dataset(opt):
    dataset = []

    vid_files = sorted(glob.glob(opt.vid_path + '/*.mp4'), key=os.path.getmtime)
    for vi, vid in enumerate(tqdm_gui(vid_files)):
    # for vid in enumerate(vid_files):
        vid_name = os.path.split(vid)[1][:-4]
        print(vid_name)

        filtered_clip_wrapper = ClipWrapper(opt.clip_filter_path, vid_name)
        video_wrapper = VideoWrapper(opt.vid_path, vid_name)
        subtitles_wrapper = SubtitleWrapper(opt.vid_path, vid_name)

        filtered_clip = filtered_clip_wrapper.get_filtered_clip()
        video = video_wrapper.get_vid()
        subtitle = subtitles_wrapper.get_subtitle()

        if video is None:
            print('[WARN] Matched video does not exist. Skip this video.')
            continue
        if filtered_clip is None:
            print('[WARN] Matched clip does not exist. Skip this video.')
            continue
        if subtitle is None:
            print('[WARN] Matched subtitle does not exist. Skip this video.')
            continue
        
        # dataset_tr.append({'vid': vid, 'clips': []})
        # dataset_val.append({'vid': vid, 'clips': []})
        # dataset_ts.append({'vid': vid, 'clips': []})

        # define current video information
        dataset.append({'vid': vid_name, 'clip_info': []})

        fps = video.get(cv2.CAP_PROP_FPS)
        for ci, clip in enumerate(filtered_clip):
            start_frame, end_frame, is_valid, landmarks = clip['clip_info'][0], clip['clip_info'][1], clip['clip_info'][2], clip['frames']            
            if is_valid:
                clip_sent_list = []
                clip_landmark_list = []
                for sub in subtitle:
                    if sub['sent'] != '':
                        sent_start_frame = second_to_frame(sub['start'], fps)
                        sent_end_frame = second_to_frame(sub['end'], fps)
                        if sent_start_frame >= start_frame and sent_end_frame <= end_frame:
                            clip_sent_list.append([sent_start_frame, sent_end_frame, sub['sent']])
                            # get local index of landmarks list
                            landmark_start_idx = sent_start_frame - start_frame
                            landmark_end_idx = sent_end_frame - start_frame
                            clip_landmark_list.append(landmarks[landmark_start_idx:landmark_end_idx])
                        
                # append clip information
                dataset[-1]['clip_info'].append({'sent': clip_sent_list,
                                                'landmarks': clip_landmark_list,
                                                'start_frame': start_frame,
                                                'end_frame': end_frame})

                print('[INFO] Current video: {}, start_frame: {}, end_frame: {}'.format(vid_name, start_frame, end_frame))
    
    count_landmarks(dataset)
    
    print('[INFO] Writing to pickle.')
    with open('{}/eye_motion_dataset.pickle'.format(opt.dataset_path), 'wb') as df:
        pickle.dump(dataset, df)

def second_to_frame(second, fps):
    return int(round(second * fps))


def count_landmarks(dataset):
    landmark_list = []
    for data in dataset:
        clip_info = data['clip_info']
        for c_info in clip_info:
            c_landmarks = c_info['landmarks']
            for landmarks in c_landmarks:
                for lm in landmarks:
                    landmark_list.append(lm)

    landmark_array = np.array(landmark_list)
    n_samples, n_features = landmark_array.shape
    print('[INFO] n_samples:{}, n_features:{}'.format(n_samples, n_features))
    # print('[INFO] Estimated running time: {:0.2f} hrs'.format(n_samples/opt.fps/60/60))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-vid_path', default='./videos')
    parser.add_argument('-facial_keypoints', default='./facial_keypoints')
    parser.add_argument('-clip_filter_path', default='./filtered_clips')
    parser.add_argument('-dataset_path', default='./dataset')
    opt = parser.parse_args()

    # make eye motion dataset
    make_dataset(opt)


if __name__ == '__main__':
    main()

