# Speech Eye Motion Dataset (SEMD)

This repository contains script to build *Speech Eye Motion Dataset*.
You can download videos and transcripts through youtube and able to extract eye skeleton.

If you have any qeustions or comments, please feel free to contact me by email ([ejhwang@nlp.kaist.ac.kr](mailto:ejhwang@nlp.kaist.ac.kr)).

## Requirement
- python 3.4+
- apiclient
- youtube_dl
- pandas
- sklearn
- tqdm
- numpy
- pickle
- cv2
- webvtt

## Usage
Before you run python code below, please make sure have following folders in your directory: 
/videos, /facial_keypoints, /clips, /dataset, /filtered_clips

### 1. Download videos from youtube.
```bash
python download_video.py -video_path ./videos/ -youtube_ch_id UC_0NfufarVw04vDfWFm8z_Q -max_result 50 -lang en -dev_key YOUR_DEV_KEY -year_from 2018 -year_to 2019
```

### 2. Extract facial landmarks.
 ```bash
python run_facial_landmarks.py -vid_path ./videos/ -facial_keypoints ./facial_keypoints -model_path ./model/shape_predictor_68_face_landmarks.dat -width 960 -height 540 -frame_threshold 500 
```

### 3. Run SceneDetect
 ```bash
python run_scenedetect.py -clip_path ./clips -vid_path ./videos
```

### 4. Run Clip Filtering
 ```bash
python run_clip_filtering.py -clip_path ./clips -vid_path ./videos -landmarks_path ./facial_keypoints -clip_filter_path ./filtered_clips -threshold 30 ratio 0.5
```

### 5. Generate Speech Eye Motion Dataset (SEMD)
 ```bash
python make_eye_motion_dataset.py -vid_path ./videos -facial_keypoints ./facial_keypoints -clip_filter_path ./filtered_clips -dataset_path ./dataset
```

### 6. Preprocess SEMD
```bash
python run_preprocessing.py dataset_path ./dataset -data_size -1 -fps 10 -n_components 7 -is_rotation_killed True
```

