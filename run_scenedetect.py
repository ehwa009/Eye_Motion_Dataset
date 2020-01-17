import argparse
import subprocess
import glob
import os
import sys

from tqdm import tqdm

def run_pyscenedetect(vid_path, opt):
    cmd = 'scenedetect --input "{}" --output "{}" -d 4 detect-content list-scenes'.format(vid_path, opt.clip_path)
    # cmd = 'scenedetect --input "{}" --output "{}" -d 4 detect-content list-scenes save-images'.format(vid_path, opt.clip_path)
    
    print('\t{}'.format(cmd))
    subprocess.run(cmd, shell=True, check=True)
    subprocess.run('exit', shell=True, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-clip_path', default='./clips')
    parser.add_argument('-vid_path', default='./videos')
    parser.add_argument('-single_iteration', default=False)
    opt = parser.parse_args()

    if not(os.path.exists(opt.clip_path)):
        os.makedirs(opt.clip_path)

    videos = glob.glob(opt.vid_path + '/*.mp4')
    print('[INFO] Total number of videos: {}'.format(str(len(videos))))

    for i, fp in tqdm(enumerate(sorted(videos, key=os.path.getmtime))):
        sys.stdout.write('{}/{}'.format(i+1, str(len(videos))))
        vid_name = os.path.split(fp)[1][-15:-4]
        csv_files = glob.glob(opt.clip_path + '/{}*.csv'.format(vid_name))
        
        if len(csv_files) > 0 and os.path.getsize(csv_files[0]):
            print('\tCSV file already exists - {}'.format(vid_name))
        else:
            run_pyscenedetect(fp, opt)
        
        if opt.single_iteration:
            break


if __name__ == '__main__':
    main()

    