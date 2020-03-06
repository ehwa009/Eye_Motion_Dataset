from __future__ import unicode_literals
from apiclient.discovery import build
from datetime import datetime, timedelta

import urllib
import argparse
import os
import youtube_dl
import sys
import glob

YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-video_path', default='./videos/')
    parser.add_argument('-youtube_ch_id', default='UC_0NfufarVw04vDfWFm8z_Q')
    parser.add_argument('-max_result', type=int, default=50)
    parser.add_argument('-lang', default='en')
    parser.add_argument('-dev_key', default='')
    parser.add_argument('-vid_idx', default=None)
    parser.add_argument('-is_only_sub', type=bool, default=False)

    # term option
    parser.add_argument('-year_from', type=int, default=2018)
    parser.add_argument('-year_to', type=int, default=2019)
    parser.add_argument('-a_month_only', default=False)
    
    opt = parser.parse_args()

    start_date = datetime(opt.year_from, 1, 1, 0, 0, 0)
    target_date = datetime(opt.year_to, 1, 1, 0, 0)

    if not os.path.exists(opt.video_path): # check the directory exists or not
        os.makedirs(opt.video_path)

    # query video indices
    vid_list = []
    try:
        IOError
        with open('video_idx.txt', 'r') as idx_file:
            print('[INFO] read video idex file from existed one.')
            for idx in idx_file:
                idx = idx.rstrip()
                if idx != '':
                    vid_list.append(idx)
    except IOError:
        print('[INFO] querying and write new video index.')
        vid_list = fetch_video_idx(opt.youtube_ch_id, start_date, target_date, opt)
        with open('video_idx.txt', 'w') as wf:
            for idx in vid_list:
                wf.write(str(idx))
                wf.write('\n')
    
    # download videos
    print('[INFO] Downloading videos.')
    vid_download(vid_list, opt)
    # delete unnecessary files
    for f in glob.glob('videos/*.en.vtt'):
        os.remove(f)
    print('[INFO] Download finished.')


def vid_download(vid_list, opt):
    ydl_opts = {'format': 'best[height=360, ext=mp4]',
                'writesubtitiels': True,
                'writeautomaticsub': True,
                'postprocessors': [{
                    'key': 'FFmpegVideoConvertor',
                    'preferedformat': 'mp4',
                }],
                'outtmpl': 'test.mp4'
                } # download options

    sub_count = 0
    cap_sub_count = 0
    total_sub_count = 0
    num_download = 0
    num_skipped = 0
    
    with open('download_log.txt', 'w') as log:
        for idx in vid_list:
            
            if opt.vid_idx:
                idx = opt.vid_idx

            ydl_opts['outtmpl'] = opt.video_path + '{}.mp4'.format(idx)
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                download_url = 'https://youtu.be/{}'.format(idx) # video url
                vid_info = ydl.extract_info(download_url, download=False)
                
                if vid_filter(vid_info):
                    try:
                        if opt.is_only_sub:
                            print('[INFO] Skip downloading video.')
                        else:
                            ydl.download([download_url])
                    except:
                        print('Error!')
                    else:
                        if vid_info.get('subtitles') != {} and vid_info.get('subtitles').get(opt.lang) != None:
                            sub_url = vid_info.get('subtitiles').get(opt.lang)[0].get('url')
                            sub_count += 1
                            total_sub_count += 1
                        
                        if vid_info.get('automatic_captions') != {}:
                            sub_url = vid_info.get('automatic_captions').get('en')[4].get('url') # 4th auto subtitle is vtt format
                            urllib.urlretrieve(sub_url, '{}{}.vtt'.format(opt.video_path, idx))
                            cap_sub_count += 1
                            total_sub_count += 1

                        log.write('{} - downloaded\n'.format(idx))
                        num_download += 1

                else:
                    log.write('{} - skipped\n'.format(idx))
                    num_skipped += 1

            print('[INFO] Downloaded: {}, Skipped: {}'.format(num_download, num_skipped))
        
        log.write('\ntotal number of subtitles: {}\n'.format(sub_count))
        log.write('downloaded: {}, skipped: {}'.format(num_download, num_skipped))



def vid_filter(info):
    passed = True
    exist_proper_format = False

    format_data = info.get('formats')
    for i in format_data:
        if i.get('height') >= 720:
            exist_proper_format = True
    if not exist_proper_format:
        passed = False
            
    if passed:
        if len(info.get('automatic_captions')) == 0 and len(info.get('subtitles')) == 0:
            passed = False
    
    return passed


def fetch_video_idx(ch_id, start_date, target_date, opt):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=opt.dev_key)
    
    start_date = start_date
    td = timedelta(days=30)
    end_date = start_date + td

    result = []

    # query 
    while start_date < target_date:
        start_string = str(start_date.isoformat()) + 'Z'
        end_string = str(end_date.isoformat()) + 'Z'
        res = youtube.search().list(part='id', 
                                    channelId=opt.youtube_ch_id, 
                                    maxResults=str(opt.max_result),
                                    publishedAfter=start_string, 
                                    publishedBefore=end_string).execute()
        result += res['items']

        while True:
            if len(res['items']) < opt.max_result or 'nextPageToken' not in res:
                break
            next_page_token = res['nextPageToken']
            res = youtube.search().list(part='id', 
                                    channelId=opt.youtube_ch_id, 
                                    maxResults=str(opt.max_result),
                                    publishedAfter=start_string, 
                                    publishedBefore=end_string,
                                    pageToken=next_page_token).execute()
            result += res['items']
        
        print('[INFO] {} to {}, number of videos found: {}'.format(start_string, end_string, len(result)))

        start_date = end_date
        end_date = start_date + td
        
        if opt.a_month_only:
            break

    # collect video idx
    vid_idx_list = []
    for vid_info in result:
        vid_id = vid_info.get('id').get('videoId')
        if vid_id is not None:
            vid_idx_list.append(vid_id)

    return vid_idx_list

if __name__ == '__main__':
    main()