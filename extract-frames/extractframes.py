# -*- coding: utf-8 -*-
"""
Created on Thu Feb 04 14:40:44 2016

@author: zyb
"""

import os

home_dir = 'E:/tv16ins/'
## get video list
file_object = open(home_dir + 'data/tv15.INS/bbc.eastenders.master.shot.reference/eastenders.collection.txt','r')
try:
    count = 0
    video_id = {}
    for eachline in file_object.readlines():
        if (len(eachline) >= 10):            
            eachline = eachline.replace("\n","")
            video_id[count] = eachline
            count += 1
    
finally:
    file_object.close( )

## extract frames (1fps)
log_file = open('extract log.txt','w')
video_home_folder = home_dir + 'data/tv15.INS/Video/'
frame_home_folder = home_dir + 'data/tv15.INS/Frame_1fps/'

for id in range(0,244):
    frame_folder = frame_home_folder+str(id)
    if os.path.exists(frame_folder)==False:
        os.mkdir(frame_folder)
        log_file.write('mksir '+frame_folder)
    cmd = 'ffmpeg -i ' + video_home_folder + video_id[id] + ' -r 1 ' + frame_folder + '/%d.1fps.png'
    log_file.write(cmd+'\n')
    os.system(cmd)

log_file.close()