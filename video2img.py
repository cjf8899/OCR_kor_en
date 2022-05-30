import cv2
import numpy as np
import os
import shutil
import argparse
parser = argparse.ArgumentParser(description='Video2img')
parser.add_argument('--video_name', default='./ocr_test.mp4', type=str, help='put your video name')
args = parser.parse_args()

save_dir = './sample_video/'+args.video_name[:-4] 
save_dir_ori = save_dir+'_ori'


if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
else:
    shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    
if not os.path.isdir(save_dir_ori):
    os.makedirs(save_dir_ori)
else:
    shutil.rmtree(save_dir_ori)
    os.makedirs(save_dir_ori)
        


cap = cv2.VideoCapture(args.video_name)
count = 0
if cap.isOpened():
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("length :", length)
    print("width :", width)
    print("height :", height)
    print("fps :", fps)
    k = np.array([
                [0,1,0],
                [1,1,1],
                [0,1,0]], np.uint8
            )

    while True:
        ret, img = cap.read()
        if ret:
            cv2.imwrite(save_dir_ori + '/%s.jpg' % str(count).zfill(5), img)
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            _,img = cv2.threshold(img,127,255, cv2.THRESH_BINARY)
            img = 255 - img

            img = cv2.erode(img, k, iterations=4)
            img = cv2.dilate(img, k, iterations=1)
            cv2.imwrite(save_dir + '/%s.jpg' % str(count).zfill(5), img)
            count += 1
#             cv2.waitKey(int(1000 / fps))
            print('save: ' + '%s.jpg' % str(count).zfill(6))
        else:
            break
else:
    print('cannot open the file')
    

    
