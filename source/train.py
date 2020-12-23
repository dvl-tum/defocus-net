#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:54:12 2018

@author: maximov
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import transforms, utils

import numpy as np
import importlib
import random
import math
from sacred import Experiment
import csv
import util_func

defocus_exp = Experiment()


@defocus_exp.config
def my_config():
    TRAIN_PARAMS = {
        'ARCH_NUM': 44,
        'FILTER_NUM': 16,
        'LEARNING_RATE': 0.0001,
        'FLAG_GPU': True,
        'EPOCHS_NUM': 1, 'EPOCH_START': 0,
        'RANDOM_LEN_INPUT': 0,
        'TRAINING_MODE': 1,

        'MODEL_STEPS': 1,

        'MODEL1_LOAD': False,
        'MODEL1_ARCH_NUM': 44,
        'MODEL1_NAME': "d01_t01", 'MODEL1_INPUT_NUM': 5,
        'MODEL1_EPOCH': 1000, 'MODEL1_FILTER_NUM': 16,
        'MODEL1_LOSS_WEIGHT': 1.,

        'MODEL2_LOAD': False,
        'MODEL2_NAME': "a44_d01_t01",
        'MODEL2_EPOCH': 500,
        'MODEL2_TRAIN_STEP': True,
    }

    DATA_PARAMS = {
        'DATA_PATH': '../data/',
        'DATA_SET': 'fs_',
        'DATA_NUM': 1,
        'FLAG_NOISE': False,
        'FLAG_SHUFFLE': False,
        'INP_IMG_NUM': 1,
        'FLAG_IO_DATA': {
            'INP_RGB': True,
            'INP_COC': False,
            'INP_AIF': False,
            'INP_DIST':True,

            'OUT_COC': True,
            'OUT_DEPTH': True,
        },
        'TRAIN_SPLIT': 0.8,
        'DATASET_SHUFFLE': True,
        'WORKERS_NUM': 4,
        'BATCH_SIZE': 16,
        'DATA_RATIO_STRATEGY': 0,
        'FOCUS_DIST': [0.1,.15,.3,0.7,1.5],
        'F_NUMBER': 1.,
        'MAX_DPT': 3.,
    }

    OUTPUT_PARAMS = {
        'RESULT_PATH': '../results/',
        'MODEL_PATH': '../models/',
        'VIZ_PORT': 8098, 'VIZ_HOSTNAME': "http://localhost", 'VIZ_ENV_NAME':'main',
        'VIZ_SHOW_INPUT': True, 'VIZ_SHOW_MID': True,
        'EXP_NUM': 1,
        'COMMENT': "Default",
    }


load_model = defocus_exp.capture(util_func.load_model)
load_data = defocus_exp.capture(util_func.load_data, prefix='DATA_PARAMS')
set_comp_device = defocus_exp.capture(util_func.set_comp_device, prefix='TRAIN_PARAMS')
set_output_folders = defocus_exp.capture(util_func.set_output_folders)


@defocus_exp.capture
def forward_pass(X, model_info, TRAIN_PARAMS, DATA_PARAMS, stacknum=1, additional_input=None):
    #to train with random number of inputs
    if TRAIN_PARAMS['RANDOM_LEN_INPUT']==1 and stacknum<DATA_PARAMS['INP_IMG_NUM']:
        X[:, model_info['inp_ch_num'] * stacknum:, :, :] = torch.zeros(
            [X.shape[0], (DATA_PARAMS['INP_IMG_NUM'] - stacknum) * model_info['inp_ch_num'], X.shape[2], X.shape[3]])

    flag_step2 = True if TRAIN_PARAMS['TRAINING_MODE']==2 else False

    outputs = model_info['model'](X, model_info['inp_ch_num'], stacknum, flag_step2=flag_step2, x2 = additional_input)

    return (outputs[1], outputs[0]) if TRAIN_PARAMS['TRAINING_MODE']==2 else (outputs, outputs)


@defocus_exp.capture
def train_model(loaders, model_info, viz_info, forward_pass, TRAIN_PARAMS, DATA_PARAMS):
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model_info['model_params'], lr=TRAIN_PARAMS['LEARNING_RATE'])

    focus_dists = DATA_PARAMS['FOCUS_DIST']

    ##### Training
    print("Total number of epochs:", TRAIN_PARAMS['EPOCHS_NUM'])
    for e_iter in range(TRAIN_PARAMS['EPOCHS_NUM'] - TRAIN_PARAMS['EPOCH_START']):
        epoch_iter = e_iter + TRAIN_PARAMS['EPOCH_START']
        loss_sum, iter_count = 0, 0

        for st_iter, sample_batch in enumerate(loaders[0]):

            # Setting up input and output data
            X = sample_batch['input'].float().to(model_info['device_comp'])
            Y = sample_batch['output'].float().to(model_info['device_comp'])
            optimizer.zero_grad()

            if TRAIN_PARAMS['TRAINING_MODE'] == 2:
                gt_step1 = Y[:, :-1, :, :]
                gt_step2 = Y[:, -1:, :, :]

            stacknum = DATA_PARAMS['INP_IMG_NUM']
            if TRAIN_PARAMS['RANDOM_LEN_INPUT'] > 0:
                stacknum = np.random.randint(1, DATA_PARAMS['INP_IMG_NUM'])
            Y = Y[:, :stacknum, :, :]

            # Focus distance maps
            X2_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
            for t in range(stacknum):
                if DATA_PARAMS['FLAG_IO_DATA']['INP_DIST']:
                    focus_distance = focus_dists[t] / focus_dists[-1]
                    X2_fcs[:, t:(t + 1), :, :] = X2_fcs[:, t:(t + 1), :, :] * (focus_distance)
            X2_fcs = X2_fcs.float().to(model_info['device_comp'])

            # Forward and compute loss
            output_step1, output_step2 = forward_pass(X, model_info, stacknum=stacknum, additional_input=X2_fcs)

            if TRAIN_PARAMS['TRAINING_MODE'] == 2:
                loss_step1, loss_step2 = 0, 0
                if DATA_PARAMS['FLAG_IO_DATA']['OUT_COC']:
                    loss_step1 = criterion(output_step1, gt_step1)
                if DATA_PARAMS['FLAG_IO_DATA']['OUT_DEPTH']:
                    loss_step2 = criterion(output_step2, gt_step2)
                loss = loss_step1 * TRAIN_PARAMS['MODEL1_LOSS_WEIGHT'] + loss_step2
            elif TRAIN_PARAMS['TRAINING_MODE'] == 1:
                loss = criterion(output_step1, Y)

            outputs = output_step2

            loss.backward()
            optimizer.step()

            # Training log
            loss_sum += loss.item()
            iter_count += 1.

            if (st_iter + 1) % 5 == 0:
                print(model_info['model_name'], 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch_iter + 1, TRAIN_PARAMS['EPOCHS_NUM'], st_iter + 1, model_info['total_steps'], loss_sum / iter_count))
                total_iter = model_info['total_steps'] * epoch_iter + st_iter

                if epoch_iter == TRAIN_PARAMS['EPOCH_START'] and (st_iter + 1) == 5:
                    viz_info.initial_viz(loss_val=loss_sum / iter_count, viz_out=outputs, viz_gt_img=gt_step2, viz_inp=X, viz_mid=output_step1)
                else:
                    viz_info.log_viz_plot(loss_val=loss_sum / iter_count, total_iter=total_iter)
                loss_sum, iter_count = 0, 0
                if (st_iter + 1) % 25 == 0:
                    viz_info.log_viz_img(viz_out=outputs, viz_gt_img=gt_step2, viz_inp=X, viz_mid=output_step1)
        #viz_info.log_viz_img(viz_out=outputs, viz_gt_img=gt_step2, viz_inp=X, viz_mid=output_step1)

        # Save model
        if (epoch_iter + 1) % 10 == 0:
            torch.save(model_info['model'].state_dict(), model_info['model_dir'] + model_info['model_name'] + '_ep' + str(0) + '.pth')


@defocus_exp.automain
def run_exp(TRAIN_PARAMS,OUTPUT_PARAMS):
    # Initial preparations
    model_dir, model_name, res_dir = set_output_folders()
    device_comp = set_comp_device()

    # Training initializations
    loaders, total_steps = load_data()
    model, inp_ch_num, out_ch_num = load_model(model_dir, model_name)
    model = model.to(device=device_comp)
    model_params = model.parameters()

    # loading weights of the first step
    if TRAIN_PARAMS['TRAINING_MODE']==2 and TRAIN_PARAMS['MODEL1_LOAD']:
        model_dir1 = OUTPUT_PARAMS['MODEL_PATH']
        model_name1 = 'a' + str(TRAIN_PARAMS['MODEL1_ARCH_NUM']).zfill(2) + '_' + TRAIN_PARAMS['MODEL1_NAME']
        print("model_name1", model_dir1, model_name1)
        pretrained_dict = torch.load( model_dir1 + model_name1+'/'+model_name1 + '_ep' + str(TRAIN_PARAMS['MODEL1_EPOCH']) + '.pth')
        model_dict = model.state_dict()
        for param_tensor in model_dict:
            for param_pre in pretrained_dict:
                if param_tensor == param_pre:
                    model_dict.update({param_tensor: pretrained_dict[param_pre]})
        model.load_state_dict(model_dict)

    if TRAIN_PARAMS['MODEL2_TRAIN_STEP'] == 2:
        model_params += list(model.parameters())

    model_info = {'model': model,
                  'model_dir': model_dir,
                  'model_name': model_name,
                  'total_steps': total_steps,
                  'inp_ch_num': inp_ch_num,
                  'out_ch_num':out_ch_num,
                  'device_comp': device_comp,
                  'model_params': model_params,
                  }
    print("inp_ch_num",inp_ch_num,"   out_ch_num",out_ch_num)

    # set visualization (optional)
    viz_info = util_func.Visualization(OUTPUT_PARAMS['VIZ_PORT'], OUTPUT_PARAMS['VIZ_HOSTNAME'], model_name, OUTPUT_PARAMS['VIZ_SHOW_INPUT'],
                                       flag_show_mid = OUTPUT_PARAMS['VIZ_SHOW_MID'], env_name=OUTPUT_PARAMS['VIZ_ENV_NAME'])
    # Run training
    train_model(loaders=loaders, model_info=model_info, viz_info=viz_info, forward_pass=forward_pass)
