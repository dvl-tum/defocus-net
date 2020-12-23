#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:54:12 2018

@author: maximov
"""

import torch, os, sys, cv2
import torch.nn as nn
from torch.nn import init
import functools
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as func
from PIL import Image

import torchvision.transforms as transforms
import numpy as np
import torch

#dynamic architecture
class RecurrentBlock(nn.Module):

	def __init__(self, input_nc, output_nc, downsampling=False, bottleneck=False, upsampling=False):
		super(RecurrentBlock, self).__init__()

		self.input_nc = input_nc
		self.output_nc = output_nc

		self.downsampling = downsampling
		self.upsampling = upsampling
		self.bottleneck = bottleneck

		self.hidden = None

		if self.downsampling:
			self.l1 = nn.Sequential(
					nn.Conv2d(input_nc, output_nc, 3, padding=1),
					nn.LeakyReLU(negative_slope=0.1)
				)
			self.l2 = nn.Sequential(
					nn.Conv2d(2 * output_nc, output_nc, 3, padding=1),
					nn.LeakyReLU(negative_slope=0.1),
					nn.Conv2d(output_nc, output_nc, 3, padding=1),
					nn.LeakyReLU(negative_slope=0.1),
				)
		elif self.upsampling:
			self.l1 = nn.Sequential(
					nn.Upsample(scale_factor=2, mode='nearest'),
					nn.Conv2d(2 * input_nc, output_nc, 3, padding=1),
					nn.LeakyReLU(negative_slope=0.1),
					nn.Conv2d(output_nc, output_nc, 3, padding=1),
					nn.LeakyReLU(negative_slope=0.1),
				)
		elif self.bottleneck:
			self.l1 = nn.Sequential(
					nn.Conv2d(input_nc, output_nc, 3, padding=1),
					nn.LeakyReLU(negative_slope=0.1)
				)
			self.l2 = nn.Sequential(
					nn.Conv2d(2 * output_nc, output_nc, 3, padding=1),
					nn.LeakyReLU(negative_slope=0.1),
					nn.Conv2d(output_nc, output_nc, 3, padding=1),
					nn.LeakyReLU(negative_slope=0.1),
				)

	def forward(self, inp):

		if self.downsampling:
			op1 = self.l1(inp)
			op2 = self.l2(torch.cat((op1, self.hidden), dim=1))

			self.hidden = op2

			return op2
		elif self.upsampling:
			op1 = self.l1(inp)

			return op1
		elif self.bottleneck:
			op1 = self.l1(inp)
			op2 = self.l2(torch.cat((op1, self.hidden), dim=1))

			self.hidden = op2

			return op2

	def reset_hidden(self, inp, dfac):
		size = list(inp.size())
		size[1] = self.output_nc
		size[2] /= dfac
		size[3] /= dfac

		for s in range(len(size)):
			size[s] = int(size[s])

		self.hidden_size = size
		self.hidden = torch.zeros(*(size)).to('cuda:0')



class RecurrentAE(nn.Module):

	def __init__(self, input_nc):
		super(RecurrentAE, self).__init__()

		self.d1 = RecurrentBlock(input_nc=input_nc, output_nc=32, downsampling=True)
		self.d2 = RecurrentBlock(input_nc=32, output_nc=43, downsampling=True)
		self.d3 = RecurrentBlock(input_nc=43, output_nc=57, downsampling=True)
		self.d4 = RecurrentBlock(input_nc=57, output_nc=76, downsampling=True)
		self.d5 = RecurrentBlock(input_nc=76, output_nc=101, downsampling=True)

		self.bottleneck = RecurrentBlock(input_nc=101, output_nc=101, bottleneck=True)

		self.u5 = RecurrentBlock(input_nc=101, output_nc=76, upsampling=True)
		self.u4 = RecurrentBlock(input_nc=76, output_nc=57, upsampling=True)
		self.u3 = RecurrentBlock(input_nc=57, output_nc=43, upsampling=True)
		self.u2 = RecurrentBlock(input_nc=43, output_nc=32, upsampling=True)
		self.u1 = RecurrentBlock(input_nc=32, output_nc=1, upsampling=True)

	def set_input(self, inp):
		self.inp = inp['A']

	def forward(self):
		d1 = func.max_pool2d(input=self.d1(self.inp), kernel_size=2)
		d2 = func.max_pool2d(input=self.d2(d1), kernel_size=2)
		d3 = func.max_pool2d(input=self.d3(d2), kernel_size=2)
		d4 = func.max_pool2d(input=self.d4(d3), kernel_size=2)
		d5 = func.max_pool2d(input=self.d5(d4), kernel_size=2)

		b = self.bottleneck(d5)

		u5 = self.u5(torch.cat((b, d5), dim=1))
		u4 = self.u4(torch.cat((u5, d4), dim=1))
		u3 = self.u3(torch.cat((u4, d3), dim=1))
		u2 = self.u2(torch.cat((u3, d2), dim=1))
		u1 = self.u1(torch.cat((u2, d1), dim=1))

		return u1

	def reset_hidden(self):
		self.d1.reset_hidden(self.inp, dfac=1)
		self.d2.reset_hidden(self.inp, dfac=2)
		self.d3.reset_hidden(self.inp, dfac=4)
		self.d4.reset_hidden(self.inp, dfac=8)
		self.d5.reset_hidden(self.inp, dfac=16)

		self.bottleneck.reset_hidden(self.inp, dfac=32)

		self.u4.reset_hidden(self.inp, dfac=16)
		self.u3.reset_hidden(self.inp, dfac=8)
		self.u5.reset_hidden(self.inp, dfac=4)
		self.u2.reset_hidden(self.inp, dfac=2)
		self.u1.reset_hidden(self.inp, dfac=1)
