import argparse
import os
import time
from glob import glob
import numpy as np
import torch
from skimage import img_as_float, img_as_int
from skimage.io import imread, imsave
import cv2

from GPBlend.gp_model_convert import GPU_model_GP_MultiStage
from gp_gan import GP_single_fusion
from gp_model import GPU_model_GP
from gp_pytorch import GP_GPU_fusion

basename = lambda path: os.path.splitext(os.path.basename(path))[0]

"""
    Note: source image, destination image and mask image have the same size.
"""


def main():
    torch.cuda.synchronize()
    torch.backends.cudnn.benchmark = True
    T0 = time.time()
    parser = argparse.ArgumentParser(description='Gaussian-Poisson GAN for high-resolution image blending')
    parser.add_argument('--invert_mask', type=int, default=0,
                        help='# the orginal mask : 1-foreground 0-background, invert mask 0-foreground 1-background')
    parser.add_argument('--inpaint_bg', type=int, default=1, help='# of base filters in encoder')
    parser.add_argument('--nef', type=int, default=64, help='# of base filters in encoder')
    parser.add_argument('--ngf', type=int, default=64, help='# of base filters in decoder or G')
    parser.add_argument('--nc', type=int, default=3, help='# of output channels in decoder or G')
    parser.add_argument('--nBottleneck', type=int, default=4000, help='# of output channels in encoder')
    parser.add_argument('--ndf', type=int, default=64, help='# of base filters in D')

    parser.add_argument('--image_size', type=int, default=64, help='The height / width of the input image to network')

    parser.add_argument('--color_weight', type=float, default=1, help='Color weight')
    parser.add_argument('--sigma', type=float, default=0.5,
                        help='Sigma for gaussian smooth of Gaussian-Poisson Equation')
    parser.add_argument('--gradient_kernel', type=str, default='normal', help='Kernel type for calc gradient')
    parser.add_argument('--smooth_sigma', type=float, default=1, help='Sigma for gaussian smooth of Laplacian pyramid')

    parser.add_argument('--supervised', type=lambda x: x == 'True', default=True,
                        help='Use unsupervised Blending GAN if False')
    parser.add_argument('--nz', type=int, default=100, help='Size of the latent z vector')
    parser.add_argument('--n_iteration', type=int, default=1000, help='# of iterations for optimizing z')

    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--g_path', default='models/blending_gan.npz', help='Path for pretrained Blending GAN model')
    parser.add_argument('--unsupervised_path', default='models/unsupervised_blending_gan.npz',
                        help='Path for pretrained unsupervised Blending GAN model')
    parser.add_argument('--list_path', default='',
                        help='File for input list in csv format: obj_path;bg_path;mask_path in each line')
    parser.add_argument('--result_folder', default='blending_result', help='Name for folder storing results')

    parser.add_argument('--src_image', default='', help='Path for source image')
    parser.add_argument('--dst_image', default='', help='Path for destination image')
    parser.add_argument('--mask_image', default='', help='Path for mask image')
    parser.add_argument('--blended_image', default='', help='Where to save blended image')

    args = parser.parse_args()

    print('Input arguments:')
    for key, value in vars(args).items():
        print('\t{}: {}'.format(key, value))
    print('')

    # Init image list
    if args.list_path:
        print('Load images from {} ...'.format(args.list_path))
        with open(args.list_path) as f:
            test_list = [line.strip().split(';') for line in f]
        print('\t {} images in total ...\n'.format(len(test_list)))
    else:
        # if os.path.isdir(args.src_image):
        #     glob('{:s}/*.jpg'.format(MASK_DIR))
        if os.path.isdir(args.src_image):
            images_path = glob('{:s}/*.png'.format(args.src_image))
            test_list = [(i, i, args.mask_image) for i in images_path]
            test_list.sort(key=lambda x: int(x[0].split('/')[-1][5:-4]))
        else:
            test_list = [(args.src_image, args.dst_image, args.mask_image)]

    if not args.blended_image:
        # Init result folder
        if not os.path.isdir(args.result_folder):
            os.makedirs(args.result_folder)
        print('Result will save to {} ...\n'.format(args.result_folder))

    mask = imread(test_list[0][2], as_gray=True).astype(float)
    if args.invert_mask:
        mask = 1 - mask
    # mask = cv2.resize(mask, (3840, 2160))
    # mask = cv2.resize(mask, (960, 540))
    mask = ((mask) > 0.5).astype(np.uint8)
    # mask = ((mask) > 0.5).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=5)
    mask = cv2.erode(mask, kernel, iterations=5)
    # mask=cv2.erode(mask, cv2.getStructuringElement(
    #     cv2.MORPH_RECT, (3, 3)), iterations=2)
    mask = ((mask) > 0.5).astype(np.uint8)

    args.gpu = 0
    total_size = len(test_list)
    if total_size > 1:
        total_size = 40  # for test time
    T_init_model = time.time()
    gp_model = GPU_model_GP(img_shape=(1, 3, *mask.shape), color_weight=args.color_weight, gpu=args.gpu)
    for param in gp_model.infer_model.parameters():
        param.grad = None
    # gp_model =GPU_model_GP_MultiStage(img_shape=(1, 3, *mask.shape), color_weight=args.color_weight, gpu=args.gpu)
    # for param in gp_model.infer_model_T1.parameters():
    #     param.grad = None
    # for param in gp_model.infer_model_T2.parameters():
    #     param.grad = None
    print('Init Time', time.time() - T_init_model, 's')
    T_infer = 0.0
    for idx in range(total_size):
        # T0_single=time.time()
        print('Processing {}/{} ...'.format(idx + 1, total_size))
        # load image....................
        T_read = time.time()
        # obj = imread(test_list[idx][0])
        obj = cv2.imread(test_list[idx][0])
        bg = cv2.imread(test_list[idx][1])[:, :, :3]

        # ----------------------无虚拟背景。需要修复---------------------------
        if args.inpaint_bg:
            bg = cv2.inpaint(bg, dilated_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            # cv2.imwrite('temp.png', bg)
        # ----------------------修复----------------------------------------
        # bg=bg*np.expand_dims(1-mask,2)
        # obj=img_as_float(obj)
        # bg = img_as_float(bg)
        print("Images Read time ", time.time() - T_read)
        # print('T0 Read + Dilate + Inpaint(option)  time', time.time() - T0_single)
        torch.cuda.synchronize()
        T2 = time.time()
        blended_ims = gp_model.GP_GPU_Model_fusion(obj, bg, mask, args.gpu)
        torch.cuda.synchronize()
        T2 = time.time() - T2
        print('T2 algorithm Time ', T2)
        T_infer += T2
        if args.blended_image:
            T3 = time.time()
            # blended_im=cv2.cvtColor(blended_im,cv2.COLOR_BGR2RGB)
            for blended_im in blended_ims:
                cv2.imwrite(args.blended_image, blended_im)
            print('T3 Save Time', time.time() - T3)
        else:
            T3 = time.time()
            for blended_im in blended_ims:
                cv2.imwrite('{}/obj_{}_bg_{}_mask_{}.png'.format(args.result_folder, basename(test_list[idx][0]),
                                                                 basename(test_list[idx][1]),
                                                                 basename(test_list[idx][2])),
                            blended_im)
            print('T3 Save Time', time.time() - T3)

    print(f'Avg Infer Time:{(T_infer) / total_size} s/f')
    print(f'Avg total Time used:{(time.time() - T0) / total_size} s/f')


if __name__ == '__main__':
    main()
