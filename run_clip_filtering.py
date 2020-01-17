import glob
import argparse
import os
import csv

from tqdm import tqdm

class EyeLandmarkWrapper:
    def __init__(self):
        pass


def read_sceneinfo(fp):
    with open(fp, 'r') as csv_file:
        frame_list = [0]
        for row in csv.reader(csv_file):
            if row:
                frame_list.append((row[1]))
        frame_list[0:3] = [] # skip header
    
    frame_list = [int(x) for x in frame_list] # str to int

    return frame_list



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-clip_path', default='./clips')
    opt = parser.parse_args()
    
    skip_flag = False

    for csv_path in tqdm(sorted(glob.glob(opt.clip_path + '/*.csv'), key=os.path.getmtime)):
        vid = os.path.split(csv_path)[1][-15:-4]
        tqdm.write(vid)

        scene_data = read_sceneinfo(csv_path)

        