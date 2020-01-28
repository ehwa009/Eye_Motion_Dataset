import cv2
import pickle
import math
import glob

from webvtt import WebVTT

def load_clip_data(clip_path):
    try:
        with open(clip_path, 'rb') as cd:
            clip_data = pickle.load(cd)
            return clip_data
    except FileNotFoundError:
        return None


class SubtitleWrapper:

    def __init__(self, vid_path, vid_name, lang):
        self.subtitle = []
        self.vid_path = vid_path
        self.vid_name = vid_name
        self.lang = lang

    def load_auto_subtitle(self):
        sub_list = glob.glob('{}/{}.vtt'.format(self.vid_path, self.vid_name))
        if len(sub_list) > 1:
            print('[WARN] There are more than one subtitle.')
            assert False
        if len(sub_list) == 1:
            for i, sub_chunk in enumerate(WebVTT().read(sub_list[0])):
                raw_sub = str(sub_chunk.raw_text)
                if raw_sub.find('/n'):
                    raw_sub = raw_sub.split('/n')


class VideoWrapper:

    def __init__(self, path, vid):
        self.vid_path = '{}/{}.mp4'.format(path, vid)
        
        self.vid = cv2.VideoCapture(self.vid_path)
        self.total_frame = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frame_rate = self.vid.get(cv2.CAP_PROP_FPS)

    def get_vid(self):
        return self.vid


class LandmarkWrapper:

    def __init__(self, path, vid):
        pickle_file = '{}/{}.pickle'.format(path, vid)
        
        with open(pickle_file, 'rb') as f:
            self.skeletons = pickle.load(f)

    def get_landmarks(self, start_frame, end_frame, interval=1):
        chunk = self.skeletons[start_frame:end_frame]
        if len(chunk) == 0:
            return []
        else:
            if interval > 1:
                return chunk[::int(interval)]
            else:
                return chunk


class ClipFilter:

    def __init__(self, vid, start_frame, end_frame, landmarks, threshold):
        self.landmarks = landmarks
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.scene_length = end_frame - start_frame
        self.vid = vid
        self.threshold = threshold

        self.filtering_result = [0, 0, 0, 0, 0, ] # [landmark_missing, picture, short, pupils, jittering, ]
        self.debugging_info = [None, None, None, ] # [landmark_missing, picture, pupils, ]
        self.msg = ''

        self.min_scene_length = 30 * 3 # assume 30 fps

    def is_landmarks_missing(self, threshold):
        n_incorrect_frame = 0

        if self.landmarks == []:
            n_incorrect_frame = self.scene_length
        else:
            for lm in self.landmarks:
                indicies = [i for i in range(4, 50)]
                if any(lm[idx] == 0 for idx in indicies):
                    n_incorrect_frame += 1
        
        self.debugging_info[0] = round(n_incorrect_frame / self.scene_length, 3)
        
        return n_incorrect_frame / self.scene_length > threshold

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
            diff += cv2.norm(frame, next_frame, cv2.NORM_L1)
            n_diff += 1
        diff /= n_diff
        
        self.debugging_info[1] = round(diff, 0)

        return diff < 3000000

    def is_too_short(self):
        return self.scene_length < self.min_scene_length

    def is_pulpil_missing(self, threshold):
        n_incorrect_frame = 0

        for lm in self.landmarks:
            indicies = [0, 1, 2, 3]
            if any(lm[idx] == 0 for idx in indicies):
                n_incorrect_frame += 1

        self.debugging_info[2] = round(n_incorrect_frame / self.scene_length, 3)

        return n_incorrect_frame / self.scene_length > threshold

    def is_too_large_jittering(self):
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
            diff += cv2.norm(frame, next_frame, cv2.NORM_L1)
            n_diff += 1
        diff /= n_diff
        
        self.debugging_info[1] = round(diff, 0)

        return diff > 100000000

    def is_long_closed_eye(self):
        pass

    def is_correct_clip(self, threshold):
        if self.is_landmarks_missing(threshold):
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

        if self.is_pulpil_missing(threshold):
            self.msg = 'There are too many missing pulpils.'
            return False
        self.filtering_result[3] = 1

        if self.is_too_large_jittering():
            self.msg = 'Can be considered there are more than a person.'
            return False
        self.filtering_result[4] = 1

        self.msg = 'Pass.'
        return True

    def get_filter_variable(self):
        return self.filtering_result, self.debugging_info, self.msg