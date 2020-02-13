import pickle

class Plot:
    def __init__(self, name):
        self.name = name


if __name__ == '__main__':
    data_path = './dataset/processed_eye_motion_dataset_pca_7.pickle'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    eye_data = data['eye_dataset']
    estimator = data['estimator']
    for ed in eye_data:
            print('[INFO] Current video: {}'.format(ed['vid']))
            for ci in ed['clip_info']:
                for sent, landmarks in zip(ci['sent'], ci['landmarks']):
                    print('TEST')
                    t_list = [i for i in range(len(landmarks))]
                    