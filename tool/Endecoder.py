from os.path import isfile, join
import cv2
import numpy as np
import os

import cv2
import os

# pathIn = '/home/sobey/Dataset/Material/Video/Material1/画质受损一般_output/Out/'
# pathIn = '../../../app/test_clip/output/'
pathIn = '/Disk1/Prosult_videjects/Blend/GP-GAN/images/car_reo'
pathIn='/Disk1/Projects/Blend/poisson/GPBlend/images/car_result_video_new_mask_6e_6'
pathIn='../GPBlend/images/1080p_11_9_V3/'
# pathOut= '../../../app/test_clip/test.mp4'
# pathOut= '../../../app/test_clip/Linear_DSTT_固定区域修复_Material冬奥会.mp4'
pathOut= '/Disk1/Projects/Blend/GP-GAN/images/car.mp4'
pathOut= '/Disk1/Projects/Blend/poisson/GPBlend/images/car_6e_6.mp4'
pathOut= '../GPBlend/images/car_1080p_11_9_V3_6e_6.mp4'
# pathOut = '/home/sobey/Dataset/Material/Video/Material1/video_output/a.avi'

def encoder():
    fps = 30
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    # for sorting the file names properly
    # print(files[0][5:][:-4])
    # files.sort(key=lambda x: int(x[:-4]))
    files.sort(key=lambda x: int(x.split('_')[1][5:]))
    print(files[0].split('_')[1][5:])
    filename = os.path.join(pathIn, files[0])
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    # out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    for i in range(len(files)):
        filename = os.path.join(pathIn , files[i])
        # reading each files
        img = cv2.imread(filename)
        out.write(img)
        if i % 30 == 0:
            print(f'Writed a new frame:{i + 1}')



    out.release()


# in_video = '/home/sobey/Dataset/Material/Video/Material1/1.1画质受损一般.ts'
# in_video = '/home/sobey/Dataset/Material/Video/Material1/中国新闻_0400_20180806035753_1_138_6500__011__high.mp4'
# in_video = '/Disk1/Video/Teleplay.mp4'
# in_video = '/Disk1/Video/老板西游记.mp4'
in_video = '/Disk1/Video/江苏新时空.mp4'
in_video='/Disk1/XR/XR Color Consistency/实验数据0929/xr-车.mp4'
def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def read_video():
    vidcap = cv2.VideoCapture(in_video)
    success, image = vidcap.read()
    count = 0
    # dir_name='./images/'
    dir_name=os.path.dirname(in_video)
    while success:
        folder = os.path.join(dir_name, os.path.basename(in_video).split('.')[0])
        create_dir(folder)
        file_path = os.path.join(folder, "frame%d.png" % count)
        cv2.imwrite(file_path, image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1


def ImagesResize(pathIn='/Disk1/XR/xr_car',pathOut='/Disk1/XR/xr_car_4k'):
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    # for sorting the file names properly
    # print(files[0][5:][:-4])
    files.sort(key=lambda x: int(x[5:-4]))
    # files.sort(key=lambda x: int(x.split('_')[1][5:]))
    # print(files[0].split('_')[1][5:])
    os.makedirs(pathOut,exist_ok=True)
    for i in range(len(files)):
        filename = os.path.join(pathIn , files[i])
        # reading each files
        img = cv2.imread(filename)
        new_img = cv2.resize(img, (3840 , 2160))
        cv2.imwrite(os.path.join(pathOut,files[i]),new_img)
        if i % 30 == 0:
            print(f'Writed a new frame:{i + 1}')


if __name__=='__main__':
    encoder()
    # read_video()
    # ImagesResize()
