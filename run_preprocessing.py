import pickle
import argparse
import pandas as pd
import numpy as np
import math

from tqdm import tqdm

CENTER_X = int(960 / 3 / 2)
CENTER_Y = int(540 / 3 / 2)

# CENTER_X = int(100 / 2)
# CENTER_Y = int(50 / 2)

def load_data(path, data_size):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if data_size < len(data):
        dataset = data[:data_size]
    else:
        dataset = data[:]
    return dataset


def save_data(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

'''
filling empty coordination, 
relocate landmark position, 
and filtering landmarks which have abnormal pulpil coordination 
'''
def run_fill_filter(eye_dataset):
    for ed in tqdm(eye_dataset):
        # preprocessing landmarks
        # print('[INFO] Current video: {}'.format(ed['vid']))
        for clip_info in ed['clip_info']:
            landmarks = clip_info['landmarks']
            filled_landmarks = []
            for landmark in landmarks:
                ci_df = pd.DataFrame(np.array(landmark))
                ci_df = ci_df.replace(0, np.nan)
                ci_df = ci_df.fillna(method='ffill') # fill NaN values in dataset
                ci_df = ci_df.rolling(3).mean() # moving average filtering
                temp_lm = []
                for landmark in ci_df.values.tolist():    
                    filled = [int(lm) for lm in landmark if not(np.isnan(lm))]
                    if len(filled) == 50:
                        # relocate center
                        diff_x = CENTER_X - filled[48]
                        diff_y = CENTER_Y - filled[49]
                        for f_i in range(0, len(filled), 2):
                            filled[f_i] += diff_x
                            filled[f_i+1] += diff_y
                        # check right pupil is outside of eye region
                        condition1 = filled[0] > filled[4] and filled[0] < filled[10]
                        condition2 = filled[1] > filled[7] and filled[1] > filled[9]
                        condition3 = filled[1] < filled[13] and filled[1] < filled[14]
                        if condition1 and condition2 and condition3:
                            temp_lm.append(filled)
                filled_landmarks.append(temp_lm)
            clip_info['landmarks'] = filled_landmarks
    
    return eye_dataset

def run_normalization(eye_dataset):
    eb_standard_len = 100

    def get_dist(x1, y1, x2, y2):
        return np.sqrt((x1-x2) ** 2 + (y1- y2) ** 2)

    def get_theta(var_x, var_y, fix_x, fix_y):
        return math.atan2(var_y - fix_y, var_x - fix_x)

    def get_new_coor(theta, dist, point):
        return dist * np.array([math.cos(theta), 
                                math.sin(theta)]) + np.array([point[0], point[1]])
    
    def run_len_norm(var_x, var_y, fix_x, fix_y, expected_len):
        angle = get_theta(var_x, var_y, fix_x, fix_y)
        new_coor = get_new_coor(angle, expected_len, [fix_x, fix_y])
        return new_coor

    for ed in tqdm(eye_dataset):
        # preprocessing landmarks
        # print('[INFO] Current video: {}'.format(ed['vid']))
        for clip_info in ed['clip_info']:
            tmp_landmarks = []
            for landmark in clip_info['landmarks']:
                tmp_landmark = []
                for lm in landmark:
                    # calculate different ratio with standard length
                    right_len_ratio = eb_standard_len / get_dist(lm[46], lm[47], lm[48], lm[49])
                    left_len_ratio = eb_standard_len / get_dist(lm[28], lm[29], lm[48], lm[49])
                    len_ratio = (right_len_ratio + left_len_ratio) / 2
                    fix_x, fix_y = lm[48], lm[49]
                    new_coor_list = []
                    for lm_i in range(0, len(lm[:48]), 2):
                        new_coor = run_len_norm(lm[lm_i], lm[lm_i+1], fix_x, fix_y,
                                            get_dist(lm[lm_i], lm[lm_i+1], fix_x, fix_y) * len_ratio)
                        new_coor_list += [int(new_coor[0]), int(new_coor[1])]
                        # pupil preprocessing
                        right_theta = get_theta(lm[0], lm[1], lm[6], lm[7])
                        right_dist = get_dist(lm[0], lm[1], lm[6], lm[7])
                        left_new_pulpil = get_new_coor(right_theta, right_dist, [lm[18], lm[19]])
                        lm[2] = int(left_new_pulpil[0])
                        lm[3] = int(left_new_pulpil[1])
                    new_coor_list += [fix_x, fix_y]
                    tmp_landmark.append(new_coor_list)                    
                tmp_landmarks.append(tmp_landmark)
            clip_info['landmarks'] = tmp_landmarks
    
    return eye_dataset



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_path', default='./dataset')
    parser.add_argument('-data_size', type=int, default=10)
    opt = parser.parse_args()

    eye_dataset = load_data('{}/eye_motion_dataset.pickle'.format(opt.dataset_path), opt.data_size)
    print('[INFO] Dataset length: {}'.format(len(eye_dataset)))
    
    print('[INFO] filling and filtering is now processing.')
    eye_dataset = run_fill_filter(eye_dataset)

    print('[INFO] Normalization is now processing.')
    eye_dataset = run_normalization(eye_dataset)
    
    # save processed dataset
    save_path = '{}/processed_eye_motion_dataset.pickle'.format(opt.dataset_path)
    print('[INFO] Save preprocessed dataset at {}'.format(save_path))
    save_data(save_path, eye_dataset)
    

if __name__ == '__main__':
    main()