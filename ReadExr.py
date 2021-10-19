import os

import numpy as np
import glob
import Imath
dir='/Disk1/XR/XR Color Consistency/实验数据0929/虚拟场景/'
filename='/Disk1/XR/XR Color Consistency/实验数据0929/虚拟场景/0000.exr'
import cv2

# cv2.imread(filename,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
im=cv2.imread(filename,-1)
im=im*65535
if (im>65535).any():
    print('HDR Sources')
im[im>65535]=65535
im=np.uint16(im)
cv2.imwrite("torus.png",im)

import OpenEXR as exr

def readEXR(filename):
    """Read color + depth data from EXR image file.

    Parameters
    ----------
    filename : str
        File path.

    Returns
    -------
    img : RGB or RGBA image in float32 format. Each color channel
          lies within the interval [0, 1].
          Color conversion from linear RGB to standard RGB is performed
          internally. See https://en.wikipedia.org/wiki/SRGB#The_forward_transformation_(CIE_XYZ_to_sRGB)
          for more information.

    Z : Depth buffer in float32 format or None if the EXR file has no Z channel.
    """

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    print(header)

    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    # convert all channels in the image to numpy arrays
    for c in header['channels']:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    colorChannels = ['R', 'G', 'B', 'A'] if 'A' in header['channels'] else ['R', 'G', 'B']
    img = np.concatenate([channelData[c][..., np.newaxis] for c in colorChannels], axis=2)

    # linear to standard RGB
    img[..., :3] = np.where(img[..., :3] <= 0.0031308,
                            12.92 * img[..., :3],
                            1.055 * np.power(img[..., :3], 1 / 2.4) - 0.055)

    # sanitize image to be in range [0, 1]
    img = np.where(img < 0.0, 0.0, np.where(img > 1.0, 1, img))

    Z = None if 'Z' not in header['channels'] else channelData['Z']

    return img, Z
out_dir='/Disk1/XR/Virtual_img/'
os.makedirs(out_dir,exist_ok=True)
files=glob.glob('{:s}/*.exr'.format(dir))
for f in files:
    porcess_img,_=readEXR(f)
    porcess_img=(porcess_img*255.0)[:,:,:3]
    cv2.imwrite(os.path.join(out_dir,f.split('/')[-1].split('.')[0]+'.png'),porcess_img)