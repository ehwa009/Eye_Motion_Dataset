import pickle
import argparse
import pandas as pd
import numpy as np

CENTER_X = int(960 / 2)
CENTER_Y = int(540 / 2)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_path', default='./dataset')
    parser.add_argument('-data_size', type=int, default=10)
    opt = parser.parse_args()

    eye_dataset = load_data('{}/eye_motion_dataset.pickle'.format(opt.dataset_path), opt.data_size)
    print('[INFO] Dataset length: {}'.format(len(eye_dataset)))
    
    for ed in eye_dataset:
        # preprocessing landmarks
        print('[INFO] Current video: {}'.format(ed['vid']))
        for clip_info in ed['clip_info']:
            landmarks = clip_info['landmarks']
            filled_landmarks = []
            for landmark in landmarks:
                ci_df = pd.DataFrame(np.array(landmark))
                ci_df = ci_df.replace(0, np.nan)
                ci_df = ci_df.fillna(method='ffill') # fill NaN values in dataset
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
    
    # save processed dataset
    save_path = '{}/processed_eye_motion_dataset.pickle'.format(opt.dataset_path)
    print('[INFO] Save preprocessed dataset at {}'.format(save_path))
    save_data(save_path, eye_dataset)
    

if __name__ == '__main__':
    main()