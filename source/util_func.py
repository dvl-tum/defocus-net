#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:54:12 2018

@author: maximov
"""

import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms, utils

from os import listdir, mkdir
from os.path import isfile, join, isdir
from visdom import Visdom
import numpy as np
import importlib
import random
import csv
import OpenEXR, Imath
from PIL import Image
from skimage import img_as_float
from skimage import measure
from scipy import stats


def _abs_val(x):
    if isinstance(x, np.ndarray) or isinstance(x, float) or isinstance(x, int):
        return np.abs(x)
    else:
        return x.abs()

# reading depth files
def read_dpt(img_dpt_path):
    # pt = Imath.PixelType(Imath.PixelType.HALF)  # FLOAT HALF
    dpt_img = OpenEXR.InputFile(img_dpt_path)
    dw = dpt_img.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    (r, g, b) = dpt_img.channels("RGB")
    dpt = np.fromstring(r, dtype=np.float16)
    dpt.shape = (size[1], size[0])
    return dpt

# to calculate circle of confusion
class CameraLens:
    def __init__(self, focal_length, sensor_size_full=(0, 0), resolution=(1, 1), aperture_diameter=None, f_number=None, depth_scale=1):
        self.focal_length = focal_length
        self.depth_scale = depth_scale
        self.sensor_size_full = sensor_size_full

        if aperture_diameter is not None:
            self.aperture_diameter = aperture_diameter
            self.f_number = (focal_length / aperture_diameter) if aperture_diameter != 0 else 0
        else:
            self.f_number = f_number
            self.aperture_diameter = focal_length / f_number

        if self.sensor_size_full is not None:
            self.resolution = resolution
            self.aspect_ratio = resolution[0] / resolution[1]
            self.sensor_size = [self.sensor_size_full[0], self.sensor_size_full[0] / self.aspect_ratio]
        else:
            self.resolution = None
            self.aspect_ratio = None
            self.sensor_size = None
            self.fov = None
            self.focal_length_pixel = None

    def _get_indep_fac(self, focus_distance):
        return (self.aperture_diameter * self.focal_length) / (focus_distance - self.focal_length)

    def get_coc(self, focus_distance, depth):
        if isinstance(focus_distance, torch.Tensor):
            for _ in range(len(depth.shape) - len(focus_distance.shape)):
                focus_distance = focus_distance.unsqueeze(-1)

        return (_abs_val(depth - focus_distance) / depth) * self._get_indep_fac(focus_distance)


class ImageDataset(torch.utils.data.Dataset):
    """Focal place dataset."""

    def __init__(self, root_dir, transform_fnc=None, flag_shuffle=False, img_num=1, data_ratio=0,
                 flag_inputs=[False, False], flag_outputs=[False, False], focus_dist=[0.1,.15,.3,0.7,1.5], f_number=0.1, max_dpt = 3.):
        self.root_dir = root_dir
        self.transform_fnc = transform_fnc
        self.flag_shuffle = flag_shuffle

        self.flag_rgb = flag_inputs[0]
        self.flag_coc = flag_inputs[1]

        self.img_num = img_num
        self.data_ratio = data_ratio

        self.flag_out_coc = flag_outputs[0]
        self.flag_out_depth = flag_outputs[1]

        self.focus_dist = focus_dist

        ##### Load and sort all images
        self.imglist_all = [f for f in listdir(root_dir) if isfile(join(root_dir, f)) and f[-7:] == "All.tif"]
        self.imglist_dpt = [f for f in listdir(root_dir) if isfile(join(root_dir, f)) and f[-7:] == "Dpt.exr"]

        print("Total number of samples", len(self.imglist_dpt), "  Total number of seqs", len(self.imglist_dpt) / img_num)

        self.imglist_all.sort()
        self.imglist_dpt.sort()

        self.camera = CameraLens(2.9 * 1e-3, f_number=f_number)
        self.max_dpt = max_dpt

    def __len__(self):
        return int(len(self.imglist_dpt))

    def __getitem__(self, idx):
        ##### Read and process an image
        idx_dpt = int(idx)
        img_dpt = read_dpt(self.root_dir + self.imglist_dpt[idx_dpt])
        img_dpt = np.clip(img_dpt, 0., self.max_dpt)
        mat_dpt = img_dpt / self.max_dpt

        mat_dpt = mat_dpt.copy()[:, :, np.newaxis]

        ind = idx * self.img_num

        num_list = list(range(self.img_num))
        if self.data_ratio == 1:
            num_list = [0, 1, 2, 3, 4]
        if self.flag_shuffle:
            random.shuffle(num_list)

        # add RGB, CoC, Depth inputs
        mats_input = np.zeros((256, 256, 0))
        mats_output = np.zeros((256, 256, 0))

        for i in range(self.img_num):
            if self.flag_rgb:
                im = Image.open(self.root_dir + self.imglist_all[ind + num_list[i]])
                img_all = np.array(im)
                mat_all = img_all.copy() / 255.
                mats_input = np.concatenate((mats_input, mat_all), axis=2)

            if self.flag_coc or self.flag_out_coc:
                img_msk = self.camera.get_coc(self.focus_dist[i], img_dpt)
                img_msk = np.clip(img_msk, 0, 1.0e-4) / 1.0e-4
                mat_msk = img_msk.copy()[:, :, np.newaxis]
                if self.flag_coc:
                    mats_input = np.concatenate((mats_input, mat_msk), axis=2)
                if self.flag_out_coc:
                    mats_output = np.concatenate((mats_output, mat_msk), axis=2)

        if self.flag_out_depth:
            mats_output = np.concatenate((mats_output, mat_dpt), axis=2)

        sample = {'input': mats_input, 'output': mats_output}

        if self.transform_fnc:
            sample = self.transform_fnc(sample)

        return sample


class ToTensor(object):
    def __call__(self, sample):
        mats_input, mats_output = sample['input'], sample['output']

        mats_input = mats_input.transpose((2, 0, 1))
        mats_output = mats_output.transpose((2, 0, 1))
        return {'input': torch.from_numpy(mats_input),
                'output': torch.from_numpy(mats_output),}


def weights_init(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal(m.weight)
        m.bias.data.fill_(0.01)


def load_data(DATA_PATH, DATA_SET, DATA_NUM, INP_IMG_NUM, FLAG_SHUFFLE, FLAG_IO_DATA, TRAIN_SPLIT,
              WORKERS_NUM, BATCH_SIZE, DATASET_SHUFFLE, DATA_RATIO_STRATEGY, FOCUS_DIST, F_NUMBER, MAX_DPT):
    data_dir = DATA_PATH + DATA_SET + str(DATA_NUM) + '/'
    img_dataset = ImageDataset(root_dir=data_dir, transform_fnc=transforms.Compose([ToTensor()]),
                               flag_shuffle=FLAG_SHUFFLE, img_num=INP_IMG_NUM, data_ratio=DATA_RATIO_STRATEGY,
                               flag_inputs=[FLAG_IO_DATA['INP_RGB'], FLAG_IO_DATA['INP_COC']],
                               flag_outputs=[FLAG_IO_DATA['OUT_COC'], FLAG_IO_DATA['OUT_DEPTH']],
                               focus_dist=FOCUS_DIST, f_number=F_NUMBER, max_dpt=MAX_DPT)

    indices = list(range(len(img_dataset)))
    split = int(len(img_dataset) * TRAIN_SPLIT)

    indices_train = indices[:split]
    indices_valid = indices[split:]

    dataset_train = torch.utils.data.Subset(img_dataset, indices_train)
    dataset_valid = torch.utils.data.Subset(img_dataset, indices_valid)

    loader_train = torch.utils.data.DataLoader(dataset=dataset_train, num_workers=WORKERS_NUM, batch_size=BATCH_SIZE, shuffle=DATASET_SHUFFLE)
    loader_valid = torch.utils.data.DataLoader(dataset=dataset_valid, num_workers=1, batch_size=1, shuffle=False)

    total_steps = int(len(dataset_train) / BATCH_SIZE)
    print("Total number of steps per epoch:", total_steps)
    print("Total number of training sample:", len(dataset_train))

    return [loader_train, loader_valid], total_steps


def load_model(model_dir, model_name, TRAIN_PARAMS, DATA_PARAMS):
    arch = importlib.import_module('arch.dofNet_arch' + str(TRAIN_PARAMS['ARCH_NUM']))

    ch_inp_num = 0
    if DATA_PARAMS['FLAG_IO_DATA']['INP_RGB']:
        ch_inp_num += 3
    if DATA_PARAMS['FLAG_IO_DATA']['INP_COC']:
        ch_inp_num += 1

    ch_out_num = 0

    if DATA_PARAMS['FLAG_IO_DATA']['OUT_DEPTH']:
        ch_out_num += 1
    ch_out_num_all = ch_out_num
    if DATA_PARAMS['FLAG_IO_DATA']['OUT_COC']:
        ch_out_num_all = ch_out_num + 1 * DATA_PARAMS['INP_IMG_NUM']
        ch_out_num += 1

    total_ch_inp = ch_inp_num * DATA_PARAMS['INP_IMG_NUM']
    if TRAIN_PARAMS['ARCH_NUM'] > 0:
        total_ch_inp = ch_inp_num

        flag_step2 = False
        if TRAIN_PARAMS['TRAINING_MODE'] == 2:
            flag_step2 = True
        model = arch.AENet(total_ch_inp, 1, TRAIN_PARAMS['FILTER_NUM'], flag_step2=flag_step2)
    else:
        model = arch.AENet(total_ch_inp, ch_out_num_all, TRAIN_PARAMS['FILTER_NUM'])
    model.apply(weights_init)

    params = list(model.parameters())
    print("model.parameters()", len(params))
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable params/Total number:",
          str(pytorch_total_params_train) + "/" + str(pytorch_total_params))

    if TRAIN_PARAMS['EPOCH_START'] > 0:
        model.load_state_dict(torch.load(model_dir + model_name + '_ep' + str(TRAIN_PARAMS['EPOCH_START']) + '.pth'))
        print("Model loaded:", model_name, " epoch:", str(TRAIN_PARAMS['EPOCH_START']))

    return model, ch_inp_num, ch_out_num


def set_comp_device(FLAG_GPU):
    device_comp = torch.device("cpu")
    if FLAG_GPU:
        device_comp = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device_comp


def set_output_folders(OUTPUT_PARAMS, DATA_PARAMS, TRAIN_PARAMS):
    model_name = 'a' + str(TRAIN_PARAMS['ARCH_NUM']).zfill(2) + '_d' + str(DATA_PARAMS['DATA_NUM']).zfill(2) + '_t' + str(
        OUTPUT_PARAMS['EXP_NUM']).zfill(2)
    res_dir = OUTPUT_PARAMS['RESULT_PATH'] + model_name + '/'
    models_dir = OUTPUT_PARAMS['MODEL_PATH'] + model_name + '/'
    if not isdir(models_dir):
        mkdir(models_dir)
    if not isdir(res_dir):
        mkdir(res_dir)
    return models_dir, model_name, res_dir


def compute_loss(Y_est, Y_gt, criterion):
    return criterion(Y_est, Y_gt)


def compute_psnr(img1, img2, mode_limit=False, msk=0):
    if mode_limit:
        msk_num = np.sum(msk)
        mse = np.sum(msk * ((img1 - img2) ** 2)) / msk_num
    else:
        mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def compute_ssim(mat_est, mat_gt, mode_limit=False, msk=0):
    ssim_full = measure.compare_ssim((mat_gt), (mat_est), data_range=img_as_float(mat_gt).max() - img_as_float(mat_gt).min(), multichannel=True,
                     full=True)
    if mode_limit:
        ssim_mean = np.sum(ssim_full[1]*msk) / (np.sum(msk))
    else:
        ssim_mean = np.sum(ssim_full[1]) / (mat_gt.shape[0] * mat_gt.shape[1] * mat_gt.shape[2])
    # dssim_mean = (1. - ssim_mean) / 2.
    return ssim_mean


def compute_pearson(a, b, mode_limit=False):
    a, b = a.flat, b.flat
    if mode_limit:
        m = np.argwhere(b > (2. / 8.))
        a = np.delete(a, m)
        b = np.delete(b, m)
    if len(a) < 10:
        coef = 0
    else:
        coef, p = stats.pearsonr(a, b)
    return coef

def compute_all_metrics(est_out, gt_out, flag_mse=True, flag_ssim=True, flag_psnr=True, flag_pearson=False, mode_limit=False):
    mat_gt = (gt_out[0]).to(torch.device("cpu")).data.numpy().transpose((1, 2, 0))
    mat_est = (est_out[0]).to(torch.device("cpu")).data.numpy().transpose((1, 2, 0))
    mat_est = np.clip(mat_est, 0., 1.)
    mse_val, ssim_val, psnr_val = 1., 0., 0.
    msk = mat_gt < 0.2
    msk_num = np.sum(msk)

    if msk_num==0:
        if flag_pearson:
            return 0, 0, 0, 0
        else:
            return 0, 0, 0

    if flag_ssim:
        ssim_val = compute_ssim(mat_gt, mat_est, mode_limit=mode_limit, msk=msk)
    if flag_psnr:
        psnr_val = compute_psnr(mat_gt, mat_est, mode_limit=mode_limit, msk=msk)
    if flag_mse:
        if mode_limit:
            mse_val = np.sum(msk*((mat_gt - mat_est) ** 2))/msk_num
        else:
            mse_val = np.mean((mat_gt - mat_est) ** 2)
    if flag_pearson:
        pearson_val = compute_pearson(mat_est, mat_gt, mode_limit=mode_limit)
        return mse_val, ssim_val, psnr_val, pearson_val
    return mse_val, ssim_val, psnr_val



# Visualize current progress
class Visualization():
    def __init__(self, port, hostname, model_name, flag_show_input=False, flag_show_mid=False, env_name='main'):
        self.viz = Visdom(port=port, server=hostname, env=env_name)
        self.loss_plot = self.viz.line(X=[0.], Y=[0.], name="train", opts=dict(title='Loss ' + model_name))
        self.flag_show_input = flag_show_input
        self.flag_show_mid = flag_show_mid

    def initial_viz(self, loss_val, viz_out, viz_gt_img, viz_inp, viz_mid):
        self.viz.line(Y=[loss_val], X=[0], win=self.loss_plot, name="train", update='replace')

        viz_out_img = torch.clamp(viz_out, 0., 1.)
        if viz_out.shape[1] > 3 or viz_out.shape[1] == 2:
            viz_out_img = viz_out_img[:, 0:1, :, :]
            viz_gt_img = viz_gt_img[:, 0:1, :, :]

        if self.flag_show_mid:
            viz_mid_img = torch.clamp(viz_mid[0, :, :, :], 0., 1.)
            viz_mid_img = viz_mid_img.unsqueeze(1)
            self.img_mid = self.viz.images(viz_mid_img, nrow=8)
        if self.flag_show_input:
            viz_inp_img = viz_inp[:, 0:3, :, :]
            self.img_input = self.viz.images(viz_inp_img, nrow=8)

        self.img_fit = self.viz.images(viz_out_img, nrow=8)
        self.img_gt = self.viz.images(viz_gt_img, nrow=8)

    def log_viz_img(self, viz_out, viz_gt_img, viz_inp, viz_mid):
        viz_out_img = torch.clamp(viz_out, 0., 1.)

        if viz_out.shape[1] > 3 or viz_out.shape[1] == 2:
            viz_out_img = viz_out_img[:, 0:1, :, :]
            viz_gt_img = viz_gt_img[:, 0:1, :, :]

        if self.flag_show_mid:
            viz_mid_img = torch.clamp(viz_mid[0, :, :, :], 0., 1.)
            viz_mid_img = viz_mid_img.unsqueeze(1)
            self.viz.images(viz_mid_img, win=self.img_mid, nrow=8)

        if self.flag_show_input:
            viz_inp_img = viz_inp[:, 0:3, :, :]
            self.viz.images(viz_inp_img, win=self.img_input, nrow=8)

        self.viz.images(viz_out_img, win=self.img_fit, nrow=8)
        self.viz.images(viz_gt_img, win=self.img_gt, nrow=8)

    def log_viz_plot(self, loss_val, total_iter):
        self.viz.line(Y=[loss_val], X=[total_iter], win=self.loss_plot, name="train", update='append')


def save_config(r, postfix="single"):
    model_name = 'a' + str(r.config['TRAIN_PARAMS']['ARCH_NUM']) + '_d' + str(r.config['DATA_PARAMS']['DATA_NUM']) + '_t' + str(
        r.config['OUTPUT_PARAMS']['EXP_NUM']).zfill(2)
    with open(r.config['OUTPUT_PARAMS']['RESULT_PATH'] + 'configs_' + postfix + '.csv', mode='a') as res_file:
        res_writer = csv.writer(res_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        res_writer.writerow([model_name, r.config['TRAIN_PARAMS'], r.config['DATA_PARAMS'], r.config['OUTPUT_PARAMS']])
