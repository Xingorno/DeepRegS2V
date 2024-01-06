import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import time
import sys
import os
import tools
from copy import copy
import numpy as np

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out

class ResNeXtBottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        
        # if stride == 2:
        #     stride = (1, 2, 2)

        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        print('stride {}'.format(stride))
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        # print("x {}".format(x.shape))
        out = self.conv1(x)
        # print("out {}".format(out.shape))
        out = self.bn1(out)
        # print("out {}".format(out.shape))
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        # print("out {}".format(out.shape))

        return out

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv1x1_3d(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3_3d(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
class ResNetBottleneck2d(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(ResNetBottleneck2d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetBottleneck3d(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(ResNetBottleneck3d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1_3d(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3_3d(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1_3d(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    


class mynet3(nn.Module):
    """ First working model! """
    def __init__(self, layers):
        self.inplanes = 64
        super(mynet3, self).__init__()
        """ Balance """
        layers = layers
        # layers = [3, 4, 6, 3]  # resnext50
        # layers = [3, 4, 23, 3]  # resnext101
        # layers = [3, 8, 36, 3]  # resnext150
        self.conv1_vol = nn.Conv3d(1, 32, kernel_size=9, stride=(2, 2, 2), padding=(4, 4, 4), bias=False)
        self.conv2_vol = nn.Conv3d(32, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        # self.conv3_vol = nn.Conv3d(128, 256, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        # self.conv4_vol = nn.Conv3d(64, 64, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.bn1_vol = nn.BatchNorm3d(32)
        self.bn2_vol = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        # self.conv_pool = nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.conv1_frame = nn.Conv3d(1, 32, kernel_size=9, stride=(2, 2, 2), padding=(4, 4, 4), bias=False)
        self.conv2_frame = nn.Conv3d(32, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        # self.conv3_frame = nn.Conv3d(128, 256, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False)

        self.conv2d_frame = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2d_frame_2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)

        self.layer1 = self._make_layer(
            ResNeXtBottleneck, 128, layers[0], shortcut_type='B', cardinality=32, stride=2)
        self.layer2 = self._make_layer(
            ResNeXtBottleneck, 256, layers[1], shortcut_type='B', cardinality=32, stride=2)
        self.layer3 = self._make_layer(
            ResNeXtBottleneck, 512, layers[2], shortcut_type='B', cardinality=32, stride=2)
        self.layer4 = self._make_layer(
            ResNeXtBottleneck, 1024, layers[3], shortcut_type='B', cardinality=32, stride=2)
        
        self.avgpool = nn.AvgPool3d((1, 4, 4), stride=2)
        self.maxpool = nn.MaxPool3d((1, 4, 7), stride=1)
        self.fc0 = nn.Linear(6*2048, 2048)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 6)

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), 
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def volBranch(self, vol):
        # print('\n********* Vol *********')
        # print('vol {}'.format(vol.shape))
        vol = self.conv1_vol(vol)
        vol = self.bn1_vol(vol)
        vol = self.relu(vol)
        # print('conv1 {}'.format(vol.shape))


        vol = self.conv2_vol(vol)
        vol = self.relu(vol)
        # print('conv2 {}'.format(vol.shape))

        return vol
    
    def frameBranch(self, frame):
        # print('\n********* Frame *********')
        # print('frame {}'.format(frame.shape))
        frame = frame.squeeze(1)
        # print('squeeze {}'.format(frame.shape))

        frame = self.conv2d_frame(frame)
        # print('conv2d_frame {}'.format(frame.shape))
        
        frame = self.conv2d_frame_2(frame)
        # print('conv2d_frame_1 {}'.format(frame.shape))

        frame = frame.unsqueeze(1)
        # print('unsqueeze {}'.format(frame.shape))

        frame = self.conv1_frame(frame)
        frame = self.bn1_vol(frame)
        frame = self.relu(frame)
        # print('conv1 {}'.format(frame.shape))

        frame = self.conv2_frame(frame)
        frame = self.relu(frame)
        # print('conv2 {}\n'.format(frame.shape))

        return frame

    def forward(self, vol, frame, initial_transform, device=None):
        input_vol = vol.clone()

        # show_size = False
        show_size = True
        if show_size:
            vol = self.volBranch(vol)
            frame = self.frameBranch(frame)

            x = torch.cat((vol, frame), 2)
            print('cat {}'.format(x.shape))
            # sys.exit()

            x = self.layer1(x)
            print('layer1 {}'.format(x.shape))

            x = self.layer2(x)
            print('layer2 {}'.format(x.shape))

            x = self.layer3(x)
            print('layer3 {}'.format(x.shape))

            x = self.layer4(x)
            print('layer4 {}'.format(x.shape))

            x = self.avgpool(x)
            print('avgpool {}'.format(x.shape))

            x = x.view(x.size(0), -1)
            print('view {}'.format(x.shape))
            x = self.fc0(x)
            print('fc0 {}'.format(x.shape))
            x = self.relu(x)

            x = self.fc1(x)
            print('fc1 {}'.format(x.shape))
            # dof_out = x.clone()
            x = self.relu(x)
            x = self.fc2(x)
            print('fc2 {}'.format(x.shape))
            # sys.exit()
            """ add the correction transformation"""
            # print('output x {}'.format(x.shape))

            # print("output x is leaf_variable (guess false): ", x.is_leaf)
            # print("output x  is required_grad (guess True): ", x.requires_grad)
            # print("device: ", device)
            # x_ = torch.zeros((1, 6))
            correction_transform = tools.dof2mat_tensor(input_dof=x).type(torch.FloatTensor).to(device)
            # print('correction_transform: ', correction_transform.device)
            # print("correction_transform is leaf_variable (guess false): ", correction_transform.is_leaf)
            # print("correction_transform is required_grad (guess True): ", correction_transform.requires_grad)

            """loading the initial transformation from (arm tracking + transformaion of 3DUS-CT/MRI)"""
            affine_transform_initial = initial_transform.type(torch.FloatTensor).to(device)
            # print('affine_transform_initial: ', affine_transform_initial)
            # print("affine_transform_initial is leaf_variable (guess True): ", affine_transform_initial.is_leaf)
            # print("affine_transform_initial is required_grad (guess False): ", affine_transform_initial.requires_grad)
            
            """get the transformation (initial + correction)"""
            affine_transform_combined = torch.matmul(correction_transform, affine_transform_initial)
            # print("affine_transform_combined: {}".format(affine_transform_combined))
            # print("affine_transform_combined: {}".format(affine_transform_combined.shape))
            # print('affine_transform_combined: ', affine_transform_combined.device)
            # print("affine_transform_combined is leaf_variable (guess false): ", affine_transform_combined.is_leaf)
            # print("affine_transform_combined is required_grad (guess True): ", affine_transform_combined.requires_grad)
            
            # affine_transform_theta = affine_transform_combined[:, 0:3, :].to(device=device)
            # print("affine_transform_theta: {}".format(affine_transform_theta.shape))
            # print('affine_transform_theta: ', affine_transform_theta.device)
            """Resampling and reslicing the volume (tested)"""
            grid_affine = F.affine_grid(theta= affine_transform_combined[:, 0:3, :], size = input_vol.shape, align_corners=True)
            # print('grid_affine size {}'.format(grid_affine.shape))
            # print("grid_affine is leaf_variable (guess false): ", grid_affine.is_leaf)
            # print("grid_affine is required_grad (guess True): ", grid_affine.requires_grad)

            # grid = grid.to(device)
            # print('grid {}'.format(grid.shape))
            vol_resampled = F.grid_sample(input_vol, grid_affine, align_corners=True)
            # print('resample {}'.format(vol_resampled.shape))
            # print('mat_out {}'.format(x.shape))
            # print("vol_resampled is leaf_variable (guess false): ", vol_resampled.is_leaf)
            # print("vol_resampled is required_grad (guess True): ", vol_resampled.requires_grad)
            
            # sys.exit()
        else:
            vol = self.volBranch(vol)
            frame = self.frameBranch(frame)

            x = torch.cat((vol, frame), 2)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            # x = self.maxpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)

            # x = torch.reshape(x, (x.shape[0], 3, 4))

            mat = tools.dof2mat_tensor(input_dof=x, device=device)
            # indices = torch.tensor([0, 1, 2]).to(device)
            # mat = torch.index_select(mat, 1, indices)

            grid = tools.myAffineGrid2(input_tensor=input_vol, input_mat=mat, 
                                       input_spacing=(1, 1, 1), device=device)

            vol_resampled = F.grid_sample(input_vol, grid)
            # del mat
            
        return vol_resampled, x
    
class RegS2Vnet_featurefusion(nn.Module):
    def __init__(self, layers):
        self.inplanes = 1
        super(RegS2Vnet_featurefusion, self).__init__()
        """ Balance """
        layers = layers
        # print("layers", layers)
        
        # self.layer1 = self._make_layer(
        #     ResNeXtBottleneck, 32, layers[0], shortcut_type='B', cardinality=32, stride=1)
        # self.layer2 = self._make_layer(
        #     ResNeXtBottleneck, 64, layers[1], shortcut_type='B', cardinality=32, stride=1)
        # self.layer3 = self._make_layer(
        #     ResNeXtBottleneck, 128, layers[2], shortcut_type='B', cardinality=32, stride=1)
        # self.layer4 = self._make_layer(
        #     ResNeXtBottleneck, 256, layers[3], shortcut_type='B', cardinality=32, stride=1)
        # self.layer5 = self._make_layer(
        #     ResNeXtBottleneck, 256, 1, shortcut_type='B', cardinality=32, stride=1)
        
        # self.avgpool = nn.AvgPool3d((3, 3, 3))
        # self.maxpool = nn.MaxPool3d((1, 3, 3))
        
        

        self.frame_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            ResNetBottleneck2d(16, 16),
            nn.MaxPool2d(4),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            ResNetBottleneck2d(32, 32),
            nn.MaxPool2d(4),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            ResNetBottleneck2d(64, 64),
            nn.MaxPool2d(3),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            ResNetBottleneck2d(64, 64),
            nn.MaxPool2d(3)
            # nn.Flatten(1,3)
        )
        self.frame_flatten = nn.Flatten(1, 3)
        self.volume_encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            ResNetBottleneck3d(16, 16),
            nn.MaxPool3d((2, 4, 4)),
            nn.Conv3d(16, 32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            ResNetBottleneck3d(32, 32),
            nn.MaxPool3d((2, 4, 4)),
            nn.Conv3d(32, 64, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            ResNetBottleneck3d(64, 64),
            nn.MaxPool3d((2, 3, 3)),
            nn.Conv3d(64, 64, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            ResNetBottleneck3d(64, 64),
            nn.MaxPool3d((2, 3, 3)),
            # nn.Flatten(3,4)
        )
        self.vol_flatten = nn.Flatten(2, 4)

        self.decoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            ResNetBottleneck3d(32, 32),
            nn.MaxPool3d((1, 3, 3)),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            ResNetBottleneck3d(64, 64),
            nn.MaxPool3d((1, 3, 3)),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            ResNetBottleneck3d(128, 128),
            nn.MaxPool3d((1, 3, 3)),
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            ResNetBottleneck3d(256, 256),
            # nn.MaxPool3d((1, 3, 3)),
            # nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # ResNetBottleneck3d(256, 256),
            nn.AvgPool3d((2, 2, 2))
        )
        self.relu = nn.ReLU(inplace=True)
        self.fc0 = nn.Linear(1536, 1024)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 6)

    # def _make_layer(self,
    #                 block,
    #                 planes,
    #                 blocks,
    #                 shortcut_type,
    #                 cardinality,
    #                 stride=1):
    #     downsample = None
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         if shortcut_type == 'A':
    #             downsample = partial(
    #                 downsample_basic_block,
    #                 planes=planes * block.expansion,
    #                 stride=stride)
    #         else:
    #             downsample = nn.Sequential(
    #                 nn.Conv3d(
    #                     self.inplanes,
    #                     planes * block.expansion,
    #                     kernel_size=1,
    #                     stride=stride,
    #                     bias=False), 
    #                 nn.BatchNorm3d(planes * block.expansion))
    #             # print("here is okay")
    #     # print("blocks: ", blocks)
    #     layers = []
    #     layers.append(block(self.inplanes, planes, cardinality, stride, downsample))
    #     self.inplanes = planes * block.expansion
    #     for i in range(1, blocks):
    #         # print("i am here")
    #         layers.append(block(self.inplanes, planes, cardinality))

    #     return nn.Sequential(*layers)

    
        

    def forward(self, vol, frame, initial_transform, device=None):
        input_vol = vol.clone()
        show_size = True
        if show_size:
            # vol = self.volBranch(vol)
            # frame = self.frameBranch(frame)
            # print('\n********* Frame encoder*********')
            # print('frame {}'.format(frame.shape))
            frame = frame.squeeze(1)
            # print('squeeze {}'.format(frame.shape))
            frame = self.frame_encoder(frame)
            # print('frame {}'.format(frame.shape))

            # print('\n********* Vol *********')
            # print('vol {}'.format(vol.shape))
            vol = self.volume_encoder(vol)
            # print('vol {}'.format(vol.shape))
            
            # """ combining the frame features and volume features"""
            # frame: batch_size X channel x h_feature x w_feature
            # vol: batch_size x channel x d_feature x h_feature x w_feature

            frame = self.frame_flatten(frame)
            # print('frame {}'.format(frame.shape))
            vol = torch.permute(vol, (0, 2, 1, 3, 4))
            # print('vol permuted shape {}'.format(vol.shape))
            
            vol = self.vol_flatten(vol)
            # print('vol shape {}'.format(vol.shape))

            x = torch.einsum('ijk, il->ijkl', vol, frame)
            # print('vol_combined_feature shape {}'.format(x.shape))
            x = x.unsqueeze(1)
            # print('vol_combined_feature {}'.format(x.shape))
            # sys.exit()
            
            """ResNeXt network (not working due to out of memory)"""
            # x = self.layer1(x)
            # print('vol_combined_feature {}'.format(x.shape))
            
            # x = self.maxpool(x)
            # print('vol_combined_feature {}'.format(x.shape))
            
            # x = self.layer2(x)
            # print('vol_combined_feature {}'.format(x.shape))
            # x = self.maxpool(x)
            # print('vol_combined_feature {}'.format(x.shape))

            # x = self.layer3(x)
            # print('vol_combined_feature {}'.format(x.shape))
            # x = self.maxpool(x)
            # print('vol_combined_feature {}'.format(x.shape))
            
            # x = self.layer4(x)
            # print('vol_combined_feature {}'.format(x.shape))
            
            # x = self.maxpool(x)
            # print('vol_combined_feature {}'.format(x.shape))

            # x = self.layer5(x)
            # print('vol_combined_feature {}'.format(x.shape))
            # # x = self.maxpool(x)
            # # print('vol_combined_feature {}'.format(x.shape))

            # x = self.avgpool(x)
            # print('avgpool {}'.format(x.shape))

            """ResNet to docoding"""
            x = self.decoder(x)
            # print('output x {}'.format(x.shape))

            x = x.view(x.size(0), -1)
            # print('view {}'.format(x.shape))
            
            
            x = self.fc0(x)
            # print('fc0 {}'.format(x.shape))
            x = self.relu(x)

            x = self.fc1(x)
            # print('fc1 {}'.format(x.shape))
        
            x = self.relu(x)
            x = self.fc2(x)
            # print('fc2 {}'.format(x.shape))
            # sys.exit()
            """ add the correction transformation"""
            # print('output x {}'.format(x.shape))


            correction_transform = tools.dof2mat_tensor(input_dof=x).type(torch.FloatTensor).to(device)
        
            # """loading the initial transformation from (arm tracking + transformaion of 3DUS-CT/MRI)"""
            # affine_transform_initial = initial_transform.type(torch.FloatTensor).to(device)
            
            # """get the transformation (initial + correction)"""
            # affine_transform_combined = torch.matmul(correction_transform, affine_transform_initial)
            
            # """Resampling and reslicing the volume (tested)"""
            # grid_affine = F.affine_grid(theta= affine_transform_combined[:, 0:3, :], size = input_vol.shape, align_corners=True)

            grid_affine = F.affine_grid(theta= correction_transform[:, 0:3, :], size = input_vol.shape, align_corners=True)
            
            vol_resampled = F.grid_sample(input_vol, grid_affine, align_corners=True)
            
            
            # sys.exit()
        
            
        return vol_resampled, x

# class mynet4(nn.Module):

#     def __init__(self, ):
#         self.inplanes = 64
#         super(mynet4, self).__init__()
#         """ Balance """
#         # layers = [3, 4, 6, 3]  # resnext50
#         # layers = [3, 4, 23, 3]  # resnext101
#         layers = [3, 8, 36, 3]  # resnext150
#         self.conv1_vol = nn.Conv3d(1, 16, kernel_size=7, stride=(1, 1, 1), padding=(3, 3, 3), bias=False)
#         self.conv2_vol = nn.Conv3d(16, 32, kernel_size=5, stride=(1, 1, 1), padding=(2, 2, 2), bias=False)
#         self.conv3_vol = nn.Conv3d(32, 64, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
#         self.mpl1_vol = nn.MaxPool3d((5, 5, 5), stride=(2, 2, 2), padding=(2, 2, 2))
#         self.mpl2_vol = nn.MaxPool3d((5, 5, 5), stride=(2, 1, 1), padding=(2, 2, 2))

#         self.bn16_vol = nn.BatchNorm3d(16)
#         self.bn32_vol = nn.BatchNorm3d(32)
#         self.bn48_vol = nn.BatchNorm3d(48)
#         self.bn64_vol = nn.BatchNorm3d(64)

#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
#         # self.conv_pool = nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

#         self.conv1_frame = nn.Conv3d(1, 16, kernel_size=7, stride=(1, 1, 1), padding=(3, 3, 3), bias=False)
#         self.conv2_frame = nn.Conv3d(16, 32, kernel_size=5, stride=(1, 1, 1), padding=(2, 2, 2), bias=False)
#         self.conv3_frame = nn.Conv3d(32, 64, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False)

#         # self.conv3_frame = nn.Conv3d(128, 256, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False)

#         self.conv2d_frame = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)

#         self.layer1 = self._make_layer(
#             ResNeXtBottleneck, 128, layers[0], shortcut_type='B', cardinality=32, stride=2)
#         self.layer2 = self._make_layer(
#             ResNeXtBottleneck, 256, layers[1], shortcut_type='B', cardinality=32, stride=2)
#         self.layer3 = self._make_layer(
#             ResNeXtBottleneck, 512, layers[2], shortcut_type='B', cardinality=32, stride=2)
#         self.layer4 = self._make_layer(
#             ResNeXtBottleneck, 1024, layers[3], shortcut_type='B', cardinality=32, stride=2)
        
#         self.avgpool = nn.AvgPool3d((1, 4, 4), stride=1)
#         self.maxpool = nn.MaxPool3d((1, 4, 7), stride=1)

#         self.fc1 = nn.Linear(2048, 512)
#         self.fc2 = nn.Linear(512, 6)

#     def _make_layer(self,
#                     block,
#                     planes,
#                     blocks,
#                     shortcut_type,
#                     cardinality,
#                     stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             if shortcut_type == 'A':
#                 downsample = partial(
#                     downsample_basic_block,
#                     planes=planes * block.expansion,
#                     stride=stride)
#             else:
#                 downsample = nn.Sequential(
#                     nn.Conv3d(
#                         self.inplanes,
#                         planes * block.expansion,
#                         kernel_size=1,
#                         stride=stride,
#                         bias=False), nn.BatchNorm3d(planes * block.expansion))

#         layers = []
#         layers.append(
#             block(self.inplanes, planes, cardinality, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes, cardinality))

#         return nn.Sequential(*layers)

#     def volBranch(self, vol):
#         # print('\n********* Vol *********')
#         # print('vol {}'.format(vol.shape))
#         vol = self.conv1_vol(vol)
#         vol = self.bn16_vol(vol)
#         vol = self.relu(vol)
#         # print('conv1 {}'.format(vol.shape))

#         vol = self.conv2_vol(vol)
#         # vol = self.bn32_vol(vol)
#         vol = self.relu(vol)
#         # print('conv2 {}'.format(vol.shape))

#         vol = self.mpl1_vol(vol)
#         # print('mpl1_vol {}'.format(vol.shape))

#         vol = self.conv3_vol(vol)
#         vol = self.bn64_vol(vol)
#         vol = self.relu(vol)
#         # print('conv3_vol {}'.format(vol.shape))

#         vol = self.mpl2_vol(vol)
#         # print('mpl2_vol {}'.format(vol.shape))

#         # sys.exit()
#         return vol
    
#     def frameBranch(self, frame):
#         # print('\n********* Frame *********')
#         # print('frame {}'.format(frame.shape))
#         frame = frame.squeeze(1)
#         # print('squeeze {}'.format(frame.shape))

#         frame = self.conv2d_frame(frame)
#         # print('conv2d_frame {}'.format(frame.shape))

#         frame = frame.unsqueeze(1)
#         # print('unsqueeze {}'.format(frame.shape))

#         frame = self.conv1_frame(frame)
#         frame = self.bn16_vol(frame)
#         frame = self.relu(frame)
#         # print('conv1_frame {}'.format(frame.shape))

#         frame = self.conv2_frame(frame)
#         # frame = self.bn32_vol(frame)
#         frame = self.relu(frame)
#         # print('conv2_frame {}'.format(frame.shape))

#         frame = self.mpl1_vol(frame)
#         # print('mpl1_vol {}'.format(frame.shape))

#         frame = self.conv3_frame(frame)
#         frame = self.bn64_vol(frame)
#         frame = self.relu(frame)
#         # print('conv3_frame {}'.format(frame.shape))

#         frame = self.mpl2_vol(frame)
#         # print('mpl2_vol {}'.format(frame.shape))

#         # sys.exit()
#         return frame

#     def forward(self, vol, frame, device=None):
#         input_vol = vol.clone()

#         show_size = False
#         # show_size = True

#         if show_size:

#             vol = self.volBranch(vol)
#             frame = self.frameBranch(frame)
#             x = torch.cat((vol, frame), 2)
#             print('\ncat {}'.format(x.shape))
#             # sys.exit()

#             x = self.layer1(x)
#             print('layer1 {}'.format(x.shape))

#             x = self.layer2(x)
#             print('layer2 {}'.format(x.shape))

#             x = self.layer3(x)
#             print('layer3 {}'.format(x.shape))

#             x = self.layer4(x)
#             print('layer4 {}'.format(x.shape))

#             x = self.avgpool(x)
#             print('avgpool {}'.format(x.shape))

#             x = x.view(x.size(0), -1)
#             print('view {}'.format(x.shape))

#             x = self.fc1(x)
#             print('fc1 {}'.format(x.shape))
#             x = self.relu(x)
#             x = self.fc2(x)
            
#             mat = tools.dof2mat_tensor(input_dof=x, device=device)
#             print('mat {}'.format(mat.shape))
            
#             print('input_vol {}'.format(input_vol.shape))
#             grid = tools.myAffineGrid2(input_tensor=input_vol, input_mat=mat, 
#                                         input_spacing=(1, 1, 1), device=device)
#             print('grid {}'.format(grid.shape))
#             vol_resampled = F.grid_sample(input_vol, grid, align_corners=True)
#             print('resample {}'.format(vol_resampled.shape))
#             print('mat_out {}'.format(x.shape))

#             sys.exit()
#         else:
#             vol = self.volBranch(vol)
#             frame = self.frameBranch(frame)

#             x = torch.cat((vol, frame), 2)

#             x = self.layer1(x)
#             x = self.layer2(x)
#             x = self.layer3(x)
#             x = self.layer4(x)

#             x = self.avgpool(x)
#             # x = self.maxpool(x)
#             x = x.view(x.size(0), -1)
#             x = self.fc1(x)
#             x = self.relu(x)
#             x = self.fc2(x)

#             # x = torch.reshape(x, (x.shape[0], 3, 4))

#             mat = tools.dof2mat_tensor(input_dof=x, device=device)
#             # indices = torch.tensor([0, 1, 2]).to(device)
#             # mat = torch.index_select(mat, 1, indices)

#             grid = tools.myAffineGrid2(input_tensor=input_vol, input_mat=mat, 
#                                        input_spacing=(1, 1, 1), device=device)
#             vol_resampled = F.grid_sample(input_vol, grid)

#         return vol_resampled, x
