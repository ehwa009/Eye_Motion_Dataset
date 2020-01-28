import os
import glob
import argparse

from tqdm import tqdm_gui
from data_utils import VideoWrapper, load_clip_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-vid_path', default='./videos')
    parser.add_argument('-facial_keypoints', default='./facial_keypoints')
    parser.add_argument('-clip_filter_path', default='./filtered_clips')
    opt = parser.parse_args()

    train = []
    validation = []
    test = []

    vid_files = sorted(glob.glob(opt.vid_path + './mp4'), key=os.path.getmtime)
    for vi, vid in enumerate(tqdm_gui(vid_files)):
        vid_name = os.path.split(vid)[1][:-4]
        print(vid_name)

        clip_data = load_clip_data('{}/{}.pickle'.format(opt.clip_filter_path, vid_name))
        if clip_data is None:
            print('[ERROR] Clip data file does not exist.')
            break

        video_wrapper = VideoWrapper(opt.vid_path, vid_name)