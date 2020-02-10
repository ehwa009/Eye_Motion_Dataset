import cv2
import pickle
import math
import glob
import re, os
import numpy as np

from webvtt import WebVTT


class ClipWrapper:

    def __init__(self, clip_path, vid_name):
        self.clip_path = clip_path
        self.vid_name = vid_name

    def get_filtered_clip(self):
        try:
            with open('{}/{}-filtered.pickle'.format(self.clip_path, self.vid_name), 'rb') as cd:
                clip_data = pickle.load(cd)
                return clip_data
        except FileNotFoundError:
            return None


class SubtitleWrapper:

    def __init__(self, vid_path, vid_name, lang='en'):
        self.vid_path = vid_path
        self.vid_name = vid_name
        self.lang = lang

    def get_subtitle(self):
        subtitle = []
        sub_list = glob.glob('{}/{}.vtt'.format(self.vid_path, self.vid_name))
        if len(sub_list) > 1:
            print('[WARN] There are more than one subtitle.')
            assert False
        if len(sub_list) == 1:
            # check wrong subtitle and rewrite vtt files
            try:
                sub_inst = WebVTT().read(sub_list[0])
            except:
                self.do_check(sub_list[0])
                sub_inst = WebVTT().read(sub_list[0])
            # iterate subtitle instance
            for i, sub_chunk in enumerate(sub_inst):
                raw_sub = str(sub_chunk.raw_text)    
                if raw_sub.find('\n'):
                    raw_sub = raw_sub.split('\n')
                else:
                    raw_sub = [raw_sub]
                sub_info = {}
                sent = ''
                for words_chunk in raw_sub:
                    words = re.sub(r"[-\".]", '', words_chunk).split(' ')
                    for word in words:
                        sent += word
                        sent += ' '
                    sub_info['sent'] = sent.strip(' ')
                    sub_info['start'] = sub_chunk.start_in_seconds
                    sub_info['end'] = sub_chunk.end_in_seconds
                subtitle.append(sub_info)
            return subtitle
        else:
            print('[ERROR] There is no subtitle file for {} video.'.format(self.vid_name))
            return None

    def do_check(self, path):
        sub_lines = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for li in range(0, len(lines)):
                if lines[li].find('<c>') != -1:
                    continue
                if lines[li] == '[Music]\n' or lines[li] == '[Music]':
                    continue
                if lines[li].find('00:') != -1 and (lines[li+1] == '\n' or lines[li+1] == ' \n'):
                    continue
                sub_lines.append(lines[li])
        
        # final check
        final = [sub_lines[0], sub_lines[1], sub_lines[2]+'\n']
        for sl in range(0, len(sub_lines), 2):
            if sub_lines[sl].find('00:') != -1:
                if sub_lines[sl+1] != '\n' or sub_lines[sl+1] != ' \n':
                    final.append(sub_lines[sl])
                    final.append(sub_lines[sl+1] + '\n')
        
        with open(path, 'w') as f:
            for filtered_sub in final:
                f.write(filtered_sub)
            

if __name__ == '__main__':
    # sub_test_path = './videos/Xo9J_G1cTsk.vtt'
    sub_test_path = './videos/h2wglfIVE0I.vtt'
    sub = SubtitleWrapper(os.path.split(sub_test_path)[0], os.path.split(sub_test_path)[1][:-4])
    subtitle = sub.get_subtitle()
    print(subtitle[:-3])
                        

class VideoWrapper:

    def __init__(self, vid_path, vid_name):

        self.vid_path = vid_path
        self.vid_name = vid_name
                
        # self.vid = cv2.VideoCapture(self.vid_path)
        # self.total_frame = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        # self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # self.frame_rate = self.vid.get(cv2.CAP_PROP_FPS)

    def get_vid(self):
        vid_full_path = '{}/{}.mp4'.format(self.vid_path, self.vid_name)
        if os.path.exists(vid_full_path):
            return cv2.VideoCapture(vid_full_path)
        else:
            return None


class LandmarkWrapper:

    def __init__(self, path, vid):
        pickle_file = '{}/{}.pickle'.format(path, vid)
        with open(pickle_file, 'rb') as f:
            self.landmarks = pickle.load(f)

    def get_landmarks(self, start_frame, end_frame, interval=1):
        chunk = self.landmarks[start_frame:end_frame]
        if len(chunk) == 0:
            return []
        else:
            if interval > 1:
                return chunk[::int(interval)]
            else:
                return chunk


class ClipFilter:

    def __init__(self, vid, start_frame, end_frame, landmarks):
        self.landmarks = landmarks
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.scene_length = end_frame - start_frame
        self.vid = vid

        self.filtering_result = [0, 0, 0, 0, 0, 0, ] # [landmark_missing, picture, short, pupils, jittering, small]
        self.debugging_info = [None, None, None, None, None, ] # [landmark_missing, picture, pupils, jittering, small]
        self.msg = ''

        self.min_scene_length = 30 * 3 # assume 30 fps

    def is_landmarks_missing(self, ratio):
        n_incorrect_frame = 0

        if self.landmarks == []:
            n_incorrect_frame = self.scene_length
        else:
            for lm in self.landmarks:
                indicies = [i for i in range(4, 50)]
                if any(lm[idx] == 0 for idx in indicies):
                    n_incorrect_frame += 1
        
        self.debugging_info[0] = round(n_incorrect_frame / self.scene_length, 3)
        
        return n_incorrect_frame / self.scene_length > ratio

    def is_picture(self):
        sampling_interval = int(math.floor(self.scene_length / 5))
        sampling_frames = list(range(self.start_frame + sampling_interval, 
                                    self.end_frame + sampling_interval + 1,
                                    sampling_interval))

        frames = []
        for frame_no in sampling_frames:
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = self.vid.read()
            frames.append(frame)

        diff = 0
        n_diff = 0
        for frame, next_frame in zip(frames, frames[1:]):
            if next_frame is not None:
                diff += cv2.norm(frame, next_frame, cv2.NORM_L1)
                n_diff += 1
        diff /= n_diff
        
        self.debugging_info[1] = round(diff, 0)

        return diff < 3000000

    def is_too_short(self):
        return self.scene_length < self.min_scene_length

    def is_pulpil_missing(self, ratio):
        n_incorrect_frame = 0

        for lm in self.landmarks:
            indicies = [0, 1, 2, 3]
            if any(lm[idx] == 0 for idx in indicies):
                n_incorrect_frame += 1

        self.debugging_info[2] = round(n_incorrect_frame / self.scene_length, 3)

        return n_incorrect_frame / self.scene_length > ratio

    def is_too_large_jittering(self): # if there is one more face detected
        sampling_interval = int(math.floor(self.scene_length / 5))
        sampling_frames = list(range(self.start_frame + sampling_interval, 
                                    self.end_frame + sampling_interval + 1,
                                    sampling_interval))

        frames = []
        for frame_no in sampling_frames:
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = self.vid.read()
            frames.append(frame)

        diff = 0
        n_diff = 0
        for frame, next_frame in zip(frames, frames[1:]):
            if next_frame is not None:
                diff += cv2.norm(frame, next_frame, cv2.NORM_L1)
                n_diff += 1
        diff /= n_diff
        
        self.debugging_info[3] = round(diff, 0)

        return diff > 100000000

    def is_landmarks_too_small(self, ratio, threshold):
        n_incorrect_frame = 0

        def get_dist(x1, y1, x2, y2):
            return np.sqrt((x1-x2) ** 2 + (y1- y2) ** 2)

        for li, landmark in enumerate(self.landmarks):
            dist = get_dist(landmark[10], landmark[11], landmark[16], landmark[17])
            if get_dist(landmark[10], landmark[11], landmark[16], landmark[17]) < threshold:
                n_incorrect_frame += 1
        self.debugging_info[4] = round(n_incorrect_frame / self.scene_length, 3)
        
        return n_incorrect_frame / self.scene_length > ratio

    def is_side(self):
        pass

    def is_correct_clip(self, ratio, threshold):
        if self.is_landmarks_missing(ratio):
            self.msg = 'There are too many missing landmarks.'
            return False
        self.filtering_result[0] = 1

        if self.is_picture():
            self.msg = 'Can be considered still picture.'
            return False
        self.filtering_result[1] = 1

        if self.is_too_short():
            self.msg = 'This clip is too short.'
            return False
        self.filtering_result[2] = 1

        if self.is_pulpil_missing(ratio):
            self.msg = 'There are too many missing pulpils.'
            return False
        self.filtering_result[3] = 1

        if self.is_too_large_jittering():
            self.msg = 'Can be considered there are more than a person.'
            return False
        self.filtering_result[4] = 1

        if self.is_landmarks_too_small(ratio, threshold):
            self.msg = 'Detected face is too small.'
            return False
        self.filtering_result[5] = 1
        

        self.msg = 'Pass.'
        return True

    def get_filter_variable(self):
        return self.filtering_result, self.debugging_info, self.msg