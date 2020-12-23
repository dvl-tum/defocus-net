#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:54:12 2018

@author: maximov
"""


import torch
import torch.nn as nn
import torch.utils.data

# static architecture
class AENet(nn.Module):

    def __init__(self,in_dim,out_dim, num_filter, n_blocks=3, flag_step2=False):
        super(AENet, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter
        self.n_blocks = n_blocks
        act_fnc = nn.LeakyReLU(0.2, inplace=True)


        self.conv_down_0 = self.convsblocks(self.in_dim, self.num_filter * 1, act_fnc)
        self.pool_0 = self.poolblock()


        for i in range(self.n_blocks):
            self.add_module('conv_down_' + str(i + 1), self.convsblocks(self.num_filter*(2**i)*2, self.num_filter*(2**i)*2, act_fnc))
            self.add_module('pool_' + str(i + 1), self.poolblock())

        self.bridge = self.convblock(self.num_filter*8*2,self.num_filter*16,act_fnc)

        for i in range(self.n_blocks+1):
            self.add_module('conv_up_' + str(i + 1), self.upconvblock(self.num_filter*(2**(3-i))*2, self.num_filter*(2**(3-i)), act_fnc) )
            self.add_module('conv_joint_' + str(i + 1), self.convblock(self.num_filter*(2**(3-i))*2,self.num_filter*(2**(3-i)),act_fnc) )

        self.conv_end = self.convblock(self.num_filter * 1, self.num_filter * 1, act_fnc)

        self.conv_out = nn.Sequential(
            nn.Conv2d(self.num_filter, self.out_dim, kernel_size=3, stride=1, padding=1),
        )

        if flag_step2:
            self.conv_down2_0 = self.convsblocks(2, self.num_filter * 1, act_fnc)
            self.pool2_0 = self.poolblock()


            for i in range(self.n_blocks):
                self.add_module('conv_down2_' + str(i + 1), self.convsblocks(self.num_filter * (2 ** i) * 2, self.num_filter * (2 ** i) * 2, act_fnc))
                self.add_module('pool2_' + str(i + 1), self.poolblock())

            self.bridge2 = self.convblock(self.num_filter * 8 * 2, self.num_filter * 16, act_fnc)

            for i in range(self.n_blocks + 1):
                self.add_module('conv_up2_' + str(i + 1),
                                self.upconvblock(self.num_filter * (2 ** (3 - i)) * 2, self.num_filter * (2 ** (3 - i)), act_fnc))
                self.add_module('conv_joint2_' + str(i + 1),
                                self.convblock(self.num_filter * (2 ** (3 - i)) * 3, self.num_filter * (2 ** (3 - i)), act_fnc))

            self.conv_end2 = self.convblock(self.num_filter * 1, self.num_filter * 1, act_fnc)

            self.conv_out2 = nn.Sequential(
                nn.Conv2d(self.num_filter, self.out_dim, kernel_size=3, stride=1, padding=1),
            )


        
    def convsblocks(self, in_ch,out_ch,act_fn):
        block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            act_fn,
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),            
            act_fn,
        )
        return block
    
    def convblock(self, in_ch,out_ch,act_fn):
        block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            act_fn,
        )
        return block
    
    def upconvblock(self,in_ch,out_ch,act_fn):
        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            act_fn,
        )
        return block
    
    def poolblock(self):
        pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        return pool


    def forward(self, x, inp=3, k=8, flag_step2=False, x2=0):
        down1 = []
        pool_temp = []
        for j in range(self.n_blocks + 1):
            down_temp = []
            for i in range(k):
                if j > 0:
                    joint_pool = torch.cat([pool_temp[0], pool_max[0]], dim=1)
                    pool_temp.pop(0)
                else:
                    joint_pool = x[:, inp * i:inp * (i + 1), :, :]

                conv = self.__getattr__('conv_down_' + str(j+0))(joint_pool)
                down_temp.append(conv)

                pool = self.__getattr__('pool_' + str(j+0))(conv)
                pool_temp.append(pool)

                pool = torch.unsqueeze(pool, 2)
                if i == 0:
                    pool_all = pool
                else:
                    pool_all = torch.cat([pool_all, pool], dim=2)
            pool_max = torch.max(pool_all, dim=2)
            down1.append(down_temp)

        bridge = []
        for i in range(k):
            join_pool = torch.cat([pool_temp[i], pool_max[0]], dim=1)
            bridge.append(self.bridge(join_pool))


        up_temp = []
        for j in range(self.n_blocks+2):
            for i in range(k):
                if j > 0:
                    joint_unpool = torch.cat([up_temp[0], down1[self.n_blocks-j+1][i]], dim=1)
                    up_temp.pop(0)
                    joint = self.__getattr__('conv_joint_' + str(j + 0))(joint_unpool)
                else:
                    joint = bridge[i]

                if j < self.n_blocks+1:
                    unpool = self.__getattr__('conv_up_' + str(j + 1))(joint)
                    up_temp.append(unpool)
                    unpool = torch.unsqueeze(unpool, 2)

                    if i == 0:
                        unpool_all = unpool
                    else:
                        unpool_all = torch.cat([unpool_all, unpool], dim=2)
                else:
                    end = self.conv_end(joint)
                    out_col = self.conv_out(end)

                    if i == 0:
                        out = out_col
                    else:
                        out = torch.cat([out, out_col], dim=1)

        if flag_step2:
            down2 = []
            pool_temp = []
            for j in range(self.n_blocks + 1):
                down_temp = []
                for i in range(k):
                    if j > 0:
                        joint_pool = torch.cat([pool_temp[0], pool_max[0]], dim=1)
                        pool_temp.pop(0)
                    else:
                        joint_pool = torch.cat([out[:, 1 * i:1 * (i + 1), :, :],x2[:, 1 * i:1 * (i + 1), :, :]], dim=1)

                    conv = self.__getattr__('conv_down2_' + str(j + 0))(joint_pool)
                    down_temp.append(conv)

                    pool = self.__getattr__('pool2_' + str(j + 0))(conv)
                    pool_temp.append(pool)

                    pool = torch.unsqueeze(pool, 2)
                    if i == 0:
                        pool_all = pool
                    else:
                        pool_all = torch.cat([pool_all, pool], dim=2)
                pool_max = torch.max(pool_all, dim=2)
                down2.append(down_temp)

            bridge = []
            for i in range(k):
                join_pool = torch.cat([pool_temp[i], pool_max[0]], dim=1)
                bridge.append(self.bridge2(join_pool))


            up_temp = []
            for j in range(self.n_blocks + 1):
                for i in range(k):
                    if j > 0:
                        joint_unpool = torch.cat([up_temp[0], unpool_max[0], down1[self.n_blocks - j + 1][i]], dim=1)
                        up_temp.pop(0)
                        joint = self.__getattr__('conv_joint2_' + str(j + 0))(joint_unpool)
                    else:
                        joint = bridge[i]

                    if j < self.n_blocks + 1:
                        unpool = self.__getattr__('conv_up2_' + str(j + 1))(joint)
                        up_temp.append(unpool)
                        unpool = torch.unsqueeze(unpool, 2)

                        if i == 0:
                            unpool_all = unpool
                        else:
                            unpool_all = torch.cat([unpool_all, unpool], dim=2)
                unpool_max = torch.max(unpool_all, dim=2)

            end2 = self.conv_end2(unpool_max[0])
            out_step2 = self.conv_out2(end2)

        if flag_step2:
            return out_step2, out
        else:
            return out

