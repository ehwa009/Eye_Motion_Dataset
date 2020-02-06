import glob
import argparse
import os
import csv
import pandas as pd
import pickle

from tqdm import tqdm
from data_utils import VideoWrapper, LandmarkWrapper, ClipFilter


def run_filtering(scene_data, landmark_wrapper, video_wrapper, opt):
    aux_info = []
    filtered_clip = []

    for i in range(scene_data.shape[0]):
        start_frame, end_frame = scene_data['Start Frame'][i], scene_data['End Frame'][i]
        landmarks_chuck = landmark_wrapper.get_landmarks(start_frame, end_frame)

        clip_filter = ClipFilter(vid=video_wrapper.get_vid(),
                                start_frame=start_frame,
                                end_frame=end_frame,
                                landmarks=landmarks_chuck)

        is_correct_clip = clip_filter.is_correct_clip(opt.ratio, opt.threshold)
        filtering_result, debug_info, msg = clip_filter.get_filter_variable()
        
        # save all clipping info
        filter_elem = {'clip_info': [start_frame, end_frame, is_correct_clip],
                        'filtering_result': filtering_result,
                        'message': msg,
                        'debug_info': debug_info}
        aux_info.append(filter_elem)

        if is_correct_clip:
            filter_elem['frames'] = landmarks_chuck
            filtered_clip.append(filter_elem)
        else:
            filter_elem['frames'] = []
            filtered_clip.append(filter_elem)

    return filtered_clip, aux_info
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-clip_path', default='./clips')
    parser.add_argument('-vid_path', default='./videos')
    parser.add_argument('-landmarks_path', default='./facial_keypoints')
    parser.add_argument('-clip_filter_path', default='./filtered_clips')

    parser.add_argument('-threshold', type=int, default=30)
    parser.add_argument('-ratio', type=int, default=0.5)
    parser.add_argument('-is_test', type=bool, default=False)
    opt = parser.parse_args()
    
    for vid_path in tqdm(sorted(glob.glob(opt.vid_path + '/*.mp4'), key=os.path.getmtime)):
        vid_name = os.path.split(vid_path)[1][:-4]
        tqdm.write('[INFO] Current video: {}'.format(vid_name))

        # make directory
        if not(os.path.exists(opt.clip_filter_path)):
            os.mkdir(opt.clip_filter_path)

        if not(os.path.exists('{}/{}-filtered.pickle'.format(opt.clip_filter_path, vid_name))):
            scene_data = pd.read_csv('{}/{}-Scenes.csv'.format(opt.clip_path, vid_name),
                                        encoding='utf-8', header=1)
            vid_data = VideoWrapper(opt.vid_path, vid_name)
            landmark_data = LandmarkWrapper(opt.landmarks_path, vid_name)

            filtered_clips, _ = run_filtering(scene_data=scene_data,
                                            landmark_wrapper=landmark_data,
                                            video_wrapper=vid_data,
                                            opt=opt)
        
            # save files
            with open('{}/{}-filtered.pickle'.format(opt.clip_filter_path, vid_name), 'wb') as cf:
                pickle.dump(filtered_clips, cf)

        if opt.is_test:
            break

if __name__ == '__main__':
    main()      

        

        

        