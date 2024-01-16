import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
from networks import resnet
from networks import resnext
import time
import os
from os import path
import random
from stl import mesh
import SimpleITK as sitk
import cv2
from datetime import datetime
import argparse
import tools
from networks import RegS2Vnet
import sys
# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from scipy import stats
import xml.etree.ElementTree as ET
import pandas as pd
import itk
from dicttoxml import dicttoxml
from xml.dom.minidom import parseString
from monai.config import print_config
from monai.data import ITKReader, ITKWriter
import matplotlib.pyplot as plt
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Spacingd,
    CenterSpatialCropd,
    MaskIntensityd,
    ScaleIntensityd,
    MapTransform,
    SpatialPadd
)

from monai.losses import (
    LocalNormalizedCrossCorrelationLoss    
)
from monai.data import DataLoader, Dataset, CacheDataset
from monai.config import KeysCollection
import matplotlib.pyplot as plt
import platform
import math
import json
from utils import loss_functions as loss_F
from utils import util_plot
# import test_network2
# import test_network
################

desc = 'Training registration generator'
parser = argparse.ArgumentParser(description=desc)

parser.add_argument('-t', '--training_mode',
                    type=str,
                    help="mode of training with different starting points",
                    default='scratch')

parser.add_argument('-m', '--model_filename',
                    type=str,
                    help="name of the pre-trained mode file",
                    default='None')

parser.add_argument('-l', '--learning_rate',
                    type=float,
                    help='Learning rate',
                    default=1e-6)

parser.add_argument('-d', '--device_no',
                    type=int,
                    choices=[0, 1, 2, 3, 4, 5, 6, 7],
                    help='GPU device number [0-7]',
                    default=0)

parser.add_argument('-e', '--epochs',
                    type=int,
                    help='number of training epochs',
                    default=500)

parser.add_argument('-b', '--batch_size',
                    type=int,
                    help='number of batch size',
                    default=4)

parser.add_argument('-n', '--network_type',
                    type=str,
                    help='choose different network architectures'
                         'the size of inputs/outputs are the same'
                         'could be original, resnext101',
                    default='mynet')


pretrain_model_str = '0213-092230'

networks3D = ['resnext50', 'resnext101', 'densenet121', 'mynet', 'mynet2', 'p3d',
              'autoencoder', 'uda']



use_last_pretrained = False
current_epoch = 0

args = parser.parse_args()
device_no = args.device_no
device = torch.device("cuda:{}".format(device_no))
batch_size = args.batch_size
num_epochs = args.epochs

project_dir = os.getcwd()
output_dir = os.path.join(project_dir, "src/outputs_DeepS2VFF_simplified_2")
isExist = os.path.exists(output_dir)
if not isExist:
    os.makedirs(output_dir)

DEEP_MODEL = True
NONDEEP_MODEL = False
RESUME_MODEL = False
TRAINED_MODEL = 4
trained_model_list = {'1': 'FVnet-supervised', '2': 'DeepS2VFF', '3':'DeepRCS2V', '4':"DeepS2VFF_simplified"}
# net = 'DeepS2VFF_refine_leakyReLU_localNCC'
net = 'DeepS2VFF_simplified'

addNoise = True

print('start device {}'.format(device))
print("network training type: {}".format(net))
print("output directory: {}".format(output_dir))
print("RESUME_MODEL : {}".format(RESUME_MODEL))
print('target training epoches: {}'.format(num_epochs))
print("training batch size: {}".format(batch_size))
print("learning rate: {}".format(args.learning_rate))
print("TRAINED_MODEL : {}".format(trained_model_list[str(TRAINED_MODEL)]))
print("adding noise to data: {}".format(addNoise))

now = datetime.now()
now_str = now.strftime('%m%d_%H%M%S')
print('now_str: {}'.format(now_str))
# create saving file
file = open(os.path.join(output_dir, '{}.txt'.format(now_str)), 'w')
file.close()

def filename_list(dir):
    images = []
    dir = os.path.expanduser(dir)
    print('dir {}'.format(dir))
    for filename in os.listdir(dir):
        # print(filename)
        file_path = path.join(dir, filename)
        images.append(file_path)
        # print(file_path)
    # print(images)
    return images


def save_info():
    file = open('infos/experiment_diary/{}.txt'.format(now_str), 'a+')
    file.write('Time_str: {}\n'.format(now_str))
    # file.write('Initial_mode: {}\n'.format(args.init_mode))
    file.write('Training_mode: {}\n'.format(args.training_mode))
    file.write('Model_filename: {}\n'.format(args.model_filename))
    file.write('Device_no: {}\n'.format(args.device_no))
    file.write('Epochs: {}\n'.format(args.epochs))
    file.write('Network_type: {}\n'.format(args.network_type))
    file.write('Learning_rate: {}\n'.format(args.learning_rate))
    # file.write('Neighbour_slices: {}\n'.format(args.neighbour_slice))
    # file.write('Infomation: {}\n'.format(args.information))
    file.write('Best_epoch: 0\n')
    file.write('Val_loss: {:.4f}\n'.format(1000))
    file.close()
    print('Information has been saved!')

def calculate_cascaded_transform(theta_list, device):
    cascaded_transform = tools.dof2mat_tensor(input_dof=torch.zeros(1, 6)).type(torch.FloatTensor)
    cascaded_transform = cascaded_transform.to(device)
    for i in range(len(theta_list)):
        trans = tools.dof2mat_tensor(input_dof=theta_list[i]).type(torch.FloatTensor)
        trans = trans.to(device)
        cascaded_transform =torch.matmul(trans, cascaded_transform)
    return cascaded_transform



def update_info(best_epoch, current_epoch, lowest_val_TRE, tv_hist):
    # readFile = open(os.path.join(output_dir, '{}.txt'.format(now_str)), "w")
    # lines = readFile.readlines()
    # readFile.close()
    
    loss_combined_train = tv_hist['train'][-1][0]
    loss_image_train = tv_hist['train'][-1][1]
    loss_dof_train = tv_hist['train'][-1][2]
    loss_combined_val = tv_hist['val'][-1][0]
    loss_image_val = tv_hist['val'][-1][1]
    loss_dof_val = tv_hist['val'][-1][2]

    file = open(os.path.join(output_dir, '{}.txt'.format(now_str)), 'a')
    file.write('Best_epoch: {}/{}, Val_loss: {:.4f}, loss_combined_train: {:.4f}, loss_image_train: {:.4f}, loss_dof_train: {:.4f}, loss_combined_val: {:.4f}, loss_image_val: {:.4f}, loss_dof_val: {:.4f}\n'.format(best_epoch, current_epoch, lowest_val_TRE, loss_combined_train, loss_image_train, loss_dof_train, loss_combined_val, loss_image_val, loss_dof_val))
    file.close()
    print('Info updated in {}!'.format(now_str))

def ConvertITKTransform2NumpyArray(transform):
    """ITK transfrom: transform_from_parent_LPS."""
    rotation = transform.GetMatrix()
    translation = transform.GetTranslation()
    center = transform.GetCenter()
    rotation_array = np.array([[rotation[0], rotation[1], rotation[2]], [rotation[3], rotation[4], rotation[5]], [rotation[6], rotation[7], rotation[8]]])
    offset = -rotation_array@center + center + translation

    transform_array = np.array([[rotation[0], rotation[1], rotation[2], offset[0]], [rotation[3], rotation[4], rotation[5], offset[1]], [rotation[6], rotation[7], rotation[8], offset[2]], [0, 0, 0, 1]])
    return transform_array

def ConvertNumpyArray2ITKTransform(transform_array):
    """ITK transform: transfrom_from_parent_LPS"""
    dimension = 3
    transform = sitk.AffineTransform(dimension)
    rotation = [transform_array[0][0], transform_array[0][1], transform_array[0][2], transform_array[1][0], transform_array[1][1], transform_array[1][2], transform_array[2][0], transform_array[2][1], transform_array[2][2]]
    translation = [transform_array[0][3], transform_array[1][3], transform_array[2][3]]
    transform.SetMatrix(rotation)
    transform.SetTranslation(translation)
    return transform

def CreateLookupTable(cases_metadata, project_dir, phase = 'train', save_flag = True):

    project_src_dir = os.path.join(project_dir, "src")
    project_data_dir = os.path.join(project_dir, "data")
    
    if phase == 'train':
        dataset_dict_fileNAME = os.path.join(project_src_dir, "training_dataset_dict.xml")
        volume_dict_fileNAME = os.path.join(project_src_dir, "training_volume_dict.xml")
    elif phase == 'val':
        dataset_dict_fileNAME = os.path.join(project_src_dir, "validation_dataset_dict.xml")
        volume_dict_fileNAME = os.path.join(project_src_dir, "validation_volume_dict.xml")
    elif phase == 'test':
        dataset_dict_fileNAME = os.path.join(project_src_dir, "test_dataset_dict.xml")
        volume_dict_fileNAME = os.path.join(project_src_dir, "test_volume_dict.xml")
    else:
        dataset_dict_fileNAME = os.path.join(project_src_dir, phase + "_dataset_dict.xml")
        volume_dict_fileNAME = os.path.join(project_src_dir, phase + "_volume_dict.xml")
    
    if platform.system() == "Linux":
        dataset_dict_fileNAME = '/'.join(dataset_dict_fileNAME.split('\\'))
        volume_dict_fileNAME = '/'.join(volume_dict_fileNAME.split('\\'))

    num_of_cases = len(cases_metadata) # number of volumes
    alldataset_LUT = []
    volume_LUT = []
    for case_index in range(0, num_of_cases):
        path = os.path.join(project_data_dir, cases_metadata[case_index].text)
        if platform.system() == 'Linux':
            path = '/'.join(path.split('\\'))
        case_tree = ET.parse(path)
        case_root = case_tree.getroot()

        # get the volume name (moving image)
        moving_image_metadata = case_root.find('moving_image')
        moving_image_fileNAME = os.path.join(project_data_dir, moving_image_metadata.find('directory').text, moving_image_metadata.find('name_raw_US_volume').text)
        if platform.system() == 'Linux':
            moving_image_fileNAME = '/'.join(moving_image_fileNAME.split('\\'))
        

        # get the mask of volume (moving image)
        moving_image_mask_metadata = case_root.find('moving_image_mask')
        moving_image_mask_fileNAME = os.path.join(project_data_dir, moving_image_mask_metadata.find('directory').text, moving_image_mask_metadata.find('name_rawdata').text)
        if platform.system() == 'Linux':
            moving_image_mask_fileNAME = '/'.join(moving_image_mask_fileNAME.split('\\'))
        

        # get the fixed image
        fixed_images_metadata = case_root.find('fixed_image')
        rangeMin = int(fixed_images_metadata.find('rangeMin').text)
        rangeMax = int(fixed_images_metadata.find('rangeMax').text)

        # get the mask of fixed image
        fixed_image_mask_metadata = case_root.find('fixed_image_mask')
        fixed_image_mask_fileNAME = os.path.join(project_data_dir, fixed_image_mask_metadata.find('directory').text, fixed_image_mask_metadata.find('name').text)
        if platform.system() == 'Linux':
            fixed_image_mask_fileNAME = '/'.join(fixed_image_mask_fileNAME.split('\\'))
        

        volume_case_dict = {'volume_ID': case_index,'volume_name': moving_image_fileNAME, "volume_mask_name": moving_image_mask_fileNAME}
        volume_LUT.append(volume_case_dict)
        # get the transform (3DUS to CT/MRI)
        tfm_3DUS_CT_metadata = case_root.find('transform_3DUS_CT')
        tfm_3DUS_CT_fileNAME = os.path.join(project_data_dir, tfm_3DUS_CT_metadata.find('directory').text, tfm_3DUS_CT_metadata.find('name').text)
        if platform.system() == 'Linux':
            tfm_3DUS_CT_fileNAME = '/'.join(tfm_3DUS_CT_fileNAME.split('\\'))

        
        # get the frame flip flag
        US_image_setting_metadata = case_root.find('US_image_setting')
        frame_flip_flag = US_image_setting_metadata.find("flip").text

        # get the transform metadata (slice to volume registration)
        tfm_RegS2V_metadata = case_root.find('slice_to_volume_registration')
        for frame_index in range(rangeMin, rangeMax + 1):
            num_letters_frame_index = len(str(frame_index))
            if num_letters_frame_index == 1:
                full_frame_index = '000' + str(frame_index)       
            elif num_letters_frame_index == 2:
                full_frame_index = '00' + str(frame_index)      
            elif num_letters_frame_index == 3:
                full_frame_index = '0' + str(frame_index)
            elif num_letters_frame_index == 4:
                full_frame_index = str(frame_index)
            else :
                print("checking out the maximum length of the filename!")
            
            # get the fixed image filename
            fixed_image_filename = "Image_" + full_frame_index + ".mha"
            fixed_image_fileNAME = os.path.join(project_data_dir, fixed_images_metadata.find('directory').text, fixed_image_filename) 
            if platform.system() == 'Linux':
                fixed_image_fileNAME = '/'.join(fixed_image_fileNAME.split('\\'))

            initial_frame_name = "initial_"+ full_frame_index + ".tfm"
            correction_frame_name = "correction_" + full_frame_index+ ".tfm"
            tfm_RegS2V_initial_fileNAME = os.path.join(project_data_dir, tfm_RegS2V_metadata.find('US_directory').text, tfm_RegS2V_metadata.find('initial_folder').text, initial_frame_name)
            tfm_RegS2V_correction_fileNAME = os.path.join(project_data_dir, tfm_RegS2V_metadata.find('US_directory').text, tfm_RegS2V_metadata.find('correction_folder').text, correction_frame_name)
            if platform.system() == 'Linux':
                tfm_RegS2V_initial_fileNAME = '/'.join(tfm_RegS2V_initial_fileNAME.split('\\'))
                tfm_RegS2V_correction_fileNAME = '/'.join(tfm_RegS2V_correction_fileNAME.split('\\'))

            if frame_index == rangeMin:
                # fixed_image_fileNAME_pre = fixed_image_fileNAME
                tfm_RegS2V_correction_fileNAME_pre = tfm_RegS2V_correction_fileNAME
            else:
                num_letters_frame_index_pre = len(str(frame_index-1))
                if num_letters_frame_index_pre == 1:
                    full_frame_index_pre = '000' + str(frame_index-1)       
                elif num_letters_frame_index_pre == 2:
                    full_frame_index_pre = '00' + str(frame_index-1)      
                elif num_letters_frame_index_pre == 3:
                    full_frame_index_pre = '0' + str(frame_index-1)
                elif num_letters_frame_index_pre == 4:
                    full_frame_index_pre = str(frame_index-1)
                else :
                    print("checking out the maximum length of the filename!")
                # fixed_image_filename_pre = "Image_" + full_frame_index_pre + ".mha"
                # fixed_image_fileNAME_pre = os.path.join(fixed_images_metadata.find('directory').text, fixed_image_filename_pre)

                correction_frame_name = "correction_" + full_frame_index_pre+ ".tfm"
                tfm_RegS2V_correction_fileNAME_pre = os.path.join(project_data_dir, tfm_RegS2V_metadata.find('US_directory').text, tfm_RegS2V_metadata.find('correction_folder').text, correction_frame_name)
                if platform.system() == 'Linux':
                    tfm_RegS2V_correction_fileNAME_pre = '/'.join(tfm_RegS2V_correction_fileNAME_pre.split('\\'))

                # case_frame_pair = [case_index, frame_index, frame_index-1]
            # case_dict = {'volume_ID': case_index, 'volume_name': moving_image_fileNAME, "volume_mask_name": moving_image_mask_fileNAME, 
            #              "frame_name":fixed_image_fileNAME, "frame_name_pre": fixed_image_fileNAME_pre, 'frame_mask_name': fixed_image_mask_fileNAME,
            #              "tfm_3DUS_CT_fileNAME": tfm_3DUS_CT_fileNAME, "tfm_RegS2V_initial_fileNAME": tfm_RegS2V_initial_fileNAME, "tfm_RegS2V_correction_fileNAME": tfm_RegS2V_correction_fileNAME, 
            #              "tfm_RegS2V_correction_fileNAME_pre": tfm_RegS2V_correction_fileNAME_pre}
            case_dict = {'volume_ID': case_index, 
                         "frame_name":fixed_image_fileNAME,'frame_mask_name': fixed_image_mask_fileNAME,
                         "tfm_RegS2V": [tfm_3DUS_CT_fileNAME, tfm_RegS2V_initial_fileNAME, tfm_RegS2V_correction_fileNAME, tfm_RegS2V_correction_fileNAME_pre],
                         "tfm_gt_diff_mat": None, "tfm_gt_diff_dof": None,
                         "tfm_RegS2V_initial_mat": None, "tfm_RegS2V_gt_mat": None, "frame_flip_flag": frame_flip_flag}
            alldataset_LUT.append(case_dict)

    data_LUT_np = np.array(alldataset_LUT)

    # print("Lookuptable: ", data_LUT_np)
    # print("size of lut:", data_LUT_np.shape)
    if save_flag:
        # dataframe = pd.DataFrame(data_LUT_np, columns=['volume case ID', 'Transform ID (gt)', 'Transform ID (initial)'])
        # dataframe.to_csv(r"E:\PROGRAM\Project_PhD\Registration\Deepcode\FVR-Net\dataset_LUT.csv")
        alldataset_xml = dicttoxml(alldataset_LUT, custom_root="all_cases")
        dom = parseString(alldataset_xml)
        dom.writexml( open(dataset_dict_fileNAME, 'w'),
               indent="\t",
               addindent="\t",
               newl='\n')
        
        volume_dict_xml = dicttoxml(volume_LUT, custom_root="volume_cases")
        dom = parseString(volume_dict_xml)
        dom.writexml( open(volume_dict_fileNAME, 'w'),
               indent="\t",
               addindent="\t",
               newl='\n')
        
        # print(dom.toprettyxml())
    return alldataset_LUT, volume_LUT

class LoadRegistrationTransformd(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False, scale = 1, volume_size = None) -> None:
        super().__init__(keys, allow_missing_keys)
        self.keys = keys
        self.scale = scale
        self.volume_size = volume_size #"W*H*D"
    def __call__(self, data):
        """Keys tfm_3DUS_CT_fileNAME: ["tfm_3DUS_CT_fileNAME", "tfm_RegS2V_initial_fileNAME", "tfm_RegS2V_correction_fileNAME", "tfm_RegS2V_correction_fileNAME_pre"]""" 
        for key in self.keys:
            tfm_3DUS_CT = sitk.ReadTransform(data[key][0]) # transform (from parent) LPS
            # print(tfm_3DUS_CT)
            tfm_3DUS_CT_np = ConvertITKTransform2NumpyArray(tfm_3DUS_CT) # from parent
            # print("tfm_3DUS_CT", np.linalg.inv(tfm_3DUS_CT_np))

            tfm_RegS2V_initial = sitk.ReadTransform(data[key][1]) # transform (from parent) LPS
            # print(tfm_RegS2V_initial)
            tfm_RegS2V_initial_np = ConvertITKTransform2NumpyArray(tfm_RegS2V_initial) # from parent
            # print("tfm_RegS2V_initial", np.linalg.inv(tfm_RegS2V_initial_np))

            tfm_RegS2V_correction = sitk.ReadTransform(data[key][2]) # transform (from parent) LPS
            tfm_RegS2V_correction_np = ConvertITKTransform2NumpyArray(tfm_RegS2V_correction) # from parent
            # print("tfm_RegS2V_correction", np.linalg.inv(tfm_RegS2V_correction_np))
            
            """add fake scaling and translation, this is for training. since the networks don't support to identify the spacing of the data array, we rescaled the data to spacing 1"""
            T_scale = np.array([[self.scale, 0, 0, 0], [0, self.scale, 0, 0], [0, 0, self.scale, 0], [0, 0, 0, 1]]) # LPS to parent
            T_scale_inv = np.linalg.inv(T_scale)
            frame_fileNAME = data["frame_name"]
            # print(frame_fileNAME)
            reader = sitk.ImageFileReader()
            reader.SetFileName(frame_fileNAME)
            reader.LoadPrivateTagsOn()
            reader.ReadImageInformation()
            spacing = reader.GetSpacing()
            size = reader.GetSize()
            T_translate = np.array([[1, 0, 0, 0], [0, 1, 0, -spacing[1]*size[1]*0.5], [0, 0, 1, 0], [0, 0, 0, 1]]) # LPS to parent
            # print(T_translate)

            """original transform_ITK (ground truth)"""
            tfm_RegS2V_gt_np_from = tfm_3DUS_CT_np@tfm_RegS2V_initial_np@tfm_RegS2V_correction_np # from parent
            tfm_RegS2V_gt_np_to = np.linalg.inv(tfm_RegS2V_gt_np_from) # (ground truth) to parent
            # print("gt transform: {}".format(tfm_RegS2V_gt_np_to))
            # print("T_translate: {}".format(T_scale_inv))
            """transform_ITK to fake the pytorch transform (rescaled and recentered the orignal one)"""
            tfm_RegS2V_gt_np_to = T_scale@T_translate@tfm_RegS2V_gt_np_to@T_scale_inv # this is the affine_transform_ITK
            # print("tfm_RegS2v_gt_np: {}".format(tfm_RegS2V_gt_np_to))
            """get transform_pytorch"""
            tfm_RegS2V_gt_pytorch = transform_conversion_ITK_to_pytorch(tfm_RegS2V_gt_np_to, self.volume_size) # this is the affine_transform_pytorch
            tfm_RegS2V_gt_mat_tensor = torch.from_numpy(tfm_RegS2V_gt_pytorch).type(torch.FloatTensor)


            """original transform_ITK (initial one)"""
            tfm_RegS2V_correction_pre = sitk.ReadTransform(data[key][3]) # transform (from parent) LPS
            tfm_RegS2V_correction_np_pre = ConvertITKTransform2NumpyArray(tfm_RegS2V_correction_pre) # from parent
            tfm_RegS2V_initial_np_from = tfm_3DUS_CT_np@tfm_RegS2V_initial_np@tfm_RegS2V_correction_np_pre # from parent
            tfm_RegS2V_initial_np_to = np.linalg.inv(tfm_RegS2V_initial_np_from) # (initial transform) to parent
            # print("initial transform:", tfm_RegS2V_initial_np_to)
            # addNoise = True
            if addNoise:
                """adding noise translation: N(mean = 0, std = 3), rotation: N(mean = 0, std = 3)"""
                translation_dof_noise = np.random.normal(0, 3, (3))
                rotation_dof_noise = np.random.normal(0, 1.5, (3))
                transform_dof_noise = np.concatenate((translation_dof_noise, rotation_dof_noise), axis =0)
                # print("transform_dof_noise:{}".format(transform_dof_noise))
                transform_noise_np = tools.dof2mat_np(transform_dof_noise) # 4 by 4 matrix
                # print("transform_noise_np:{}".format(transform_noise_np))
                """transform_ITK to fake the pytorch transform (rescaled and recentered the orignal one)"""
                tfm_RegS2V_initial_np_to = T_scale@transform_noise_np@T_translate@tfm_RegS2V_initial_np_to@T_scale_inv  # this is the affine_transform_ITK
            else:
                """transform_ITK to fake the pytorch transform (rescaled and recentered the orignal one)"""
                tfm_RegS2V_initial_np_to = T_scale@T_translate@tfm_RegS2V_initial_np_to@T_scale_inv  # this is the affine_transform_ITK


            # """transform_ITK to fake the pytorch transform (rescaled and recentered the orignal one)"""
            # tfm_RegS2V_initial_np_to = T_scale@T_translate@tfm_RegS2V_initial_np_to@T_scale_inv  # this is the affine_transform_ITK
            """get transform_pytorch"""
            tfm_RegS2V_initial_pytorch = transform_conversion_ITK_to_pytorch(tfm_RegS2V_initial_np_to, self.volume_size) # this is the affine_transform_pytorch
            tfm_RegS2V_initial_mat_tensor = torch.from_numpy(tfm_RegS2V_initial_pytorch).type(torch.FloatTensor)
            """caculate the different between the initial one and ground truth, which is for training"""

            # # """this is based on the transform_ITK, we decided to use the transform_pytorch to train our model, see below"""
            # tfm_gt_diff_mat = tfm_RegS2V_gt_np_to@np.linalg.inv(tfm_RegS2V_initial_np_to)
            # tfm_gt_diff_mat_tensor = torch.from_numpy(tfm_gt_diff_mat)
            # tfm_gt_diff_dof = tools.mat2dof_np(input_mat=tfm_gt_diff_mat)
            # tfm_gt_diff_dof_tensor = torch.from_numpy(tfm_gt_diff_dof[:6])
            # # tfm_RegS2V_initial_mat_tensor = torch.from_numpy(tfm_RegS2V_initial_np_to)
            """Note: converting to ITK transform needs to be inversed"""
            tfm_gt_diff_mat_normalized = tfm_RegS2V_gt_pytorch@np.linalg.inv(tfm_RegS2V_initial_pytorch) # 
            # tfm_gt_diff_mat_normalized = np.linalg.inv(tfm_RegS2V_initial_pytorch) @ tfm_RegS2V_gt_pytorch# need to be inversed for normalized transform
            tfm_gt_diff_mat_tensor_normalized = torch.from_numpy(tfm_gt_diff_mat_normalized).type(torch.FloatTensor)
            tfm_gt_diff_dof_rad_normalized = tools.mat2dof_tensor(tfm_gt_diff_mat_tensor_normalized, degree = 'rad')
            # dof_deg = tools.mat2dof_tensor(tfm_gt_diff_mat_tensor_normalized, degree = 'deg')
            
            # matrix = tools.dof2mat_tensor_normalized(dof_rad)
            #################test########################
            # print("mat_tensor: ", tfm_gt_diff_mat_tensor_normalized)
            # print("mat_tensor * ini: ", tfm_gt_diff_mat_tensor_normalized @ tfm_RegS2V_initial_mat_tensor)
            # print("gt_tensor: ", tfm_RegS2V_gt_mat_tensor)

           

            # tfm_gt_diff_mat = tfm_RegS2V_gt_np_to @ np.linalg.inv(tfm_RegS2V_initial_np_to) 
            # tfm_gt_diff_mat_tensor = torch.from_numpy(tfm_gt_diff_mat)
            # # print("diff_gt_mat (1): {}".format(tfm_gt_diff_mat_tensor))

            # # vol_size = [200, 160, 120]
            # correction_transform_ITK_LPS = transform_conversion_pytorch_to_tfm_ITK_LPS(tfm_gt_diff_mat_tensor_normalized, self.volume_size)
            # dof_est_unnormalized = tools.mat2dof_tensor(correction_transform_ITK_LPS, degree = 'rad')

            # print("dof_est_unnormalized(1): {}".format(dof_est_unnormalized))
            # print("dof_est_unnormalized(2): {}".format(tools.mat2dof_tensor(tfm_gt_diff_mat_tensor, degree = 'rad')))

            # # print("tfm_gt_diff_mat:{}".format(tfm_gt_diff_mat))
            # tfm_gt_diff_dof_normalized = tools.mat2dof_np(input_mat=tfm_gt_diff_mat_normalized)
            # tfm_gt_diff_dof_tensor_normalized = torch.from_numpy(tfm_gt_diff_dof_normalized[:6])
            # print("tfm_gt_diff_dof_tensor_normalized: ", tfm_gt_diff_dof_tensor_normalized)


            # print("dof_est_unnormalized:{}".format(dof_est_unnormalized))
            # print("tfm_gt_diff_dof_tensor_normalized:{}".format(tfm_gt_diff_dof_tensor_normalized))
            # print("diff_gt_mat(2): {}".format(correction_transform_ITK_LPS))

            data["tfm_gt_diff_mat"] = tfm_gt_diff_mat_tensor_normalized # difference normlized by affine_grid
            data["tfm_gt_diff_dof"] = tfm_gt_diff_dof_rad_normalized # difference normlized by affine_grid
            data["tfm_RegS2V_initial_mat"] = tfm_RegS2V_initial_mat_tensor # difference normlized by affine_grid
            data["tfm_RegS2V_gt_mat"] = tfm_RegS2V_gt_mat_tensor # difference normlized by affine_grid


        return data

def transform_conversion_pytorch_to_ITK(affine_transform_pytorch_initial, affine_transform_pytorch_correction, volume_size):
    #note: affine_transform_ITK (to parent)
    """ to calucate the affine_transorm_ITK, when you get the output from our network"""
    """our network is to cacluate the correction transform, assuming we already have a good iniital transform (affine_transform_ITK_initial)"""
    affine_transform_pytorch = affine_transform_pytorch_correction@affine_transform_pytorch_initial
    T_normalized = np.array([[2/volume_size[0], 0, 0, 0], [0, 2/volume_size[1], 0, 0], [0, 0, 2/volume_size[2], 0], [0, 0, 0, 1]]) # LPS to parent
    affine_transform_ITK = np.linalg.inv(T_normalized) @ affine_transform_pytorch @ T_normalized

    return affine_transform_ITK

def transform_conversion_pytorch_to_original_ITK(affine_transform_pytorch_initial, affine_transform_pytorch_correction, volume_size, frame_spacing, frame_size, scale):
    # TODO: need to test the correctness
    #note: affine_transform_ITK (to parent); affine_transform_original_ITK (to parent)
    """ to calucate the affine_transorm_ITK, when you get the output from our network"""
    """our network is to cacluate the correction transform, assuming we already have a good iniital transform (affine_transform_ITK_initial)"""
    affine_transform_pytorch = affine_transform_pytorch_correction@affine_transform_pytorch_initial
    T_normalized = np.array([[2/volume_size[0], 0, 0, 0], [0, 2/volume_size[1], 0, 0], [0, 0, 2/volume_size[2], 0], [0, 0, 0, 1]]) # LPS to parent
    affine_transform_ITK = np.linalg.inv(T_normalized) @ affine_transform_pytorch @ T_normalized
    
    T_scale = np.array([[scale, 0, 0, 0], [0, scale, 0, 0], [0, 0, scale, 0], [0, 0, 0, 1]]) # LPS to parent
    T_scale_inv = np.linalg.inv(T_scale)
    
    T_translate = np.array([[1, 0, 0, 0], [0, 1, 0, -frame_spacing[1]*frame_size[1]*0.5], [0, 0, 1, 0], [0, 0, 0, 1]]) # LPS to parent
    T_translate_inv = np.linalg.inv(T_translate)
    #tfm_RegS2V_gt_np_to = T_scale@T_translate@tfm_RegS2V_gt_np_to@T_scale_inv
    affine_transform_original_ITK = T_translate_inv@T_scale_inv@affine_transform_ITK@T_scale

    return affine_transform_original_ITK

def transform_conversion_dof_normalized_to_ITK(dof_normalized, vol_size, device):
    mat_normalized = tools.dof2mat_tensor_normalized(dof_normalized)
    mat_normalized = mat_normalized.to(device)
    transform_ITK_LPS = transform_conversion_pytorch_to_tfm_ITK_LPS(mat_normalized, vol_size, device=device)
    dof_ITK = tools.mat2dof_tensor(transform_ITK_LPS, degree = 'deg')
    dof_ITK = dof_ITK.to(device)
    return dof_ITK

def transform_conversion_ITK_to_pytorch(affine_transform_ITK, volume_size):
    # note: affine_transform_ITK (to parent)
    """to achieve the conversion of transform from the ITK transformation to pytorch transformation"""
    """For the input and output objects, both of them are rescaled to spacing (1mm, 1mm, 1mm) and recentered the volume, which is to fake the pytorch transform"""
    """For the pytorch, the input and output are the torches, which dont have the spacing information, so we can treat them as 1."""
    """To understand how the transform_ITK and transform_pytorch can work well and know the relationship, we need to bring anther package (Scipy) to help us"""
    """The conversion workflow is: transform_ITK -> transform_scipy -> transform_pytorch"""
    """For the scipy, the input and outout dont have the spacing information, which are the same as the pytorch transform. Differently, the input and output are represented as [1, 2,..., N_w] by [1, 2, ..., N_h] by [1, 2, ..., N_d].
    And the transform_scipy is applied around the first voxel [0, 0, 0] by default. If the volume center is set as the (0, 0, 0) in spatial domain, we need to translate the transform_ITK to the center of volume first T_translate, 
    then apply the transform_ITK, after that, we need to translate back the object."""
    """For the pytorch, the transformation is similiar as the transform_scipy. Differently, the input and ouput need to be normalized first. That means that instead of representing by [1, 2,..., N_w] by [1, 2, ..., N_h] by [1, 2, ..., N_d], 
    the input and output are represented by [-1,,..., 0, ..., 1] by [-1,,..., 0, ..., 1] by [-1,,..., 0, ..., 1]. For the numpy array input, the grid_sample will recoginize them as 
    [-N_w/2, ..., -2, -1, 0, 1, 2, ..., N_w/2] by [-N_h/2, ..., -2, -1, 0, 1, 2, ..., N_h/2] by [-N_d/2, ..., -2, -1, 0, 1, 2, ..., N_d/2]"""
    
    # print(image_array_volume)
    # affine_transform_ITK = data_2DUS[21]["tfm_RegS2V_gt_mat"].numpy()
    # affine_transform = np.array([[np.cos(45*np.pi/180), -np.sin(45*np.pi/180), 0, 0], [np.sin(45*np.pi/180), np.cos(45*np.pi/180), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) # LPS to parent

    
    affine_transform_inverse = np.linalg.inv(affine_transform_ITK) # for the grid_sampling, the transform is inversed
    """transform_scipy -> transform_pytorch"""
    # T_normalized = np.array([[2/volume_size[0], 0, 0, -1], [0, 2/volume_size[1], 0, -1], [0, 0, 2/volume_size[2], -1], [0, 0, 0, 1]]) # LPS to parent
    T_normalized = np.array([[2/volume_size[0], 0, 0, 0], [0, 2/volume_size[1], 0, 0], [0, 0, 2/volume_size[2], 0], [0, 0, 0, 1]]) # LPS to parent
    affine_transform_pytorch = T_normalized @ affine_transform_inverse @ np.linalg.inv(T_normalized)
    
    return affine_transform_pytorch # 4 by 4

def transform_conversion_ITK_torch_to_pytorch(affine_transform_ITK_torch, volume_size):
    # note: affine_transform_ITK (to parent)
    """to achieve the conversion of transform from the ITK transformation to pytorch transformation"""
    """For the input and output objects, both of them are rescaled to spacing (1mm, 1mm, 1mm) and recentered the volume, which is to fake the pytorch transform"""
    """For the pytorch, the input and output are the tensors, which don't have the spacing information, so we can treat them as 1."""
    """To understand how the transform_ITK and transform_pytorch can work well and know the relationship, we need to bring anther package (Scipy) to help us"""
    """The conversion workflow is: transform_ITK -> transform_scipy -> transform_pytorch"""
    """For the scipy, the input and outout dont have the spacing information neither, which are the same as the pytorch transform. Differently, the input and output are represented as [1, 2,..., N_w] by [1, 2, ..., N_h] by [1, 2, ..., N_d].
    And the transform_scipy is applied around the first voxel [0, 0, 0] by default. If the volume center is set as the (0, 0, 0) in spatial domain, we need to translate the transform_ITK to the center of volume first T_translate, 
    then apply the transform_ITK, after that, we need to translate back the object."""
    """For the pytorch, the transformation is similiar as the transform_scipy. Differently, the input and ouput need to be normalized first. That means that instead of representing by [1, 2,..., N_w] by [1, 2, ..., N_h] by [1, 2, ..., N_d], 
    the input and output are represented by [-1,,..., 0, ..., 1] by [-1,,..., 0, ..., 1] by [-1,,..., 0, ..., 1]. For the numpy array input, the grid_sample will recoginize them as 
    [-N_w/2, ..., -2, -1, 0, 1, 2, ..., N_w/2] by [-N_h/2, ..., -2, -1, 0, 1, 2, ..., N_h/2] by [-N_d/2, ..., -2, -1, 0, 1, 2, ..., N_d/2]"""
    
    # print(image_array_volume)
    # affine_transform_ITK = data_2DUS[21]["tfm_RegS2V_gt_mat"].numpy()
    # affine_transform = np.array([[np.cos(45*np.pi/180), -np.sin(45*np.pi/180), 0, 0], [np.sin(45*np.pi/180), np.cos(45*np.pi/180), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) # LPS to parent

    affine_transform_inverse = torch.linalg.inv(affine_transform_ITK_torch)
    # affine_transform_inverse = np.linalg.inv(affine_transform_ITK) # for the grid_sampling, the transform is inversed
    """transform_scipy -> transform_pytorch"""
    # T_normalized = np.array([[2/volume_size[0], 0, 0, -1], [0, 2/volume_size[1], 0, -1], [0, 0, 2/volume_size[2], -1], [0, 0, 0, 1]]) # LPS to parent
    # T_normalized = np.array([[2/volume_size[0], 0, 0, 0], [0, 2/volume_size[1], 0, 0], [0, 0, 2/volume_size[2], 0], [0, 0, 0, 1]]) # LPS to parent
    T_normalized = torch.tensor([[2/volume_size[0], 0.0, 0.0, 0.0], [0.0, 2/volume_size[1], 0.0, 0.0], [0.0, 0.0, 2/volume_size[2], 0.0], [0.0, 0.0, 0.0, 1.0]]).type(torch.FloatTensor)
    # affine_transform_pytorch = T_normalized @ affine_transform_inverse @ np.linalg.inv(T_normalized)
    affine_transform_pytorch = T_normalized @ affine_transform_inverse @ torch.linalg.inv(T_normalized)
    return affine_transform_pytorch # 4 by 4

def transform_conversion_pytorch_to_tfm_ITK_LPS(tranform_normalized, volume_size, device = "cpu"):
    
    T_normalized = torch.tensor([[2/volume_size[0], 0.0, 0.0, 0.0], [0.0, 2/volume_size[1], 0.0, 0.0], [0.0, 0.0, 2/volume_size[2], 0.0], [0.0, 0.0, 0.0, 1.0]]).type(torch.FloatTensor)
    T_normalized = T_normalized.to(device)
    affine_transform_inverse = torch.linalg.inv(T_normalized) @ tranform_normalized @ T_normalized
    
    affine_transform_ITK_LPS = torch.linalg.inv(affine_transform_inverse)
    return affine_transform_ITK_LPS

def transform_conversion_ITK_to_scipy(affine_transform_ITK, volume_size):
    #note: affine_transform_ITK (to parent)
    """to achieve the conversion of transform from the ITK transformation to pytorch transformation"""
    """For the input and output objects, both of them are rescaled to spacing (1mm, 1mm, 1mm) and recentered the volume, which is to fake the pytorch transform"""
    """For the pytorch, the input and output are the torches, which dont have the spacing information, so we can treat them as 1."""
    """To understand how the transform_ITK and transform_pytorch can work well and know the relationship, we need to bring anther package (Scipy) to help us"""
    """The conversion workflow is: transform_ITK -> transform_scipy -> transform_pytorch"""
    """For the scipy, the input and outout dont have the spacing information, which are the same as the pytorch transform. Differently, the input and output are represented as [1, 2,..., N_w] by [1, 2, ..., N_h] by [1, 2, ..., N_d].
    And the transform_scipy is applied around the first voxel [0, 0, 0] by default. If the volume center is set as the (0, 0, 0) in spatial domain, we need to translate the transform_ITK to the center of volume first T_translate, 
    then apply the transform_ITK, after that, we need to translate back the object."""
    """For the pytorch, the transformation is similiar as the transform_scipy. Differently, the input and ouput need to be normalized first. That means that instead of representing by [1, 2,..., N_w] by [1, 2, ..., N_h] by [1, 2, ..., N_d], 
    the input and output are represented by [-1,,..., 0, ..., 1] by [-1,,..., 0, ..., 1] by [-1,,..., 0, ..., 1]. Therefore, based on transform_scipy, we need to renormalized the input first, then apply transform_scipy, after that re-normalized back the ojects"""
    
    # print(image_array_volume)
    # affine_transform_ITK = data_2DUS[21]["tfm_RegS2V_gt_mat"].numpy()
    # affine_transform = np.array([[np.cos(45*np.pi/180), -np.sin(45*np.pi/180), 0, 0], [np.sin(45*np.pi/180), np.cos(45*np.pi/180), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) # LPS to parent

    """transform_ITK -> transform_scipy"""
    T_translate = np.array([[1, 0, 0, -volume_size[0]*0.5], [0, 1, 0, -volume_size[1]*0.5], [0, 0, 1, -volume_size[2]*0.5], [0, 0, 0, 1]]) # LPS to parent
    affine_transform_scipy = np.linalg.inv(T_translate)@affine_transform_ITK@T_translate

    return affine_transform_scipy # 4 by 4

def train_model(model, training_dataset_frame, training_dataset_volume, validation_dataset_frame, validation_dateset_volume, num_cases):
    since = time.time()

    loss_mse = nn.MSELoss()
    # loss_localNCC = LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size= 13, kernel_type="rectangular", reduction="mean")
    # loss_localNCC = loss_F.LocalNCC(device = device, kernel_size =(71, 71), stride=(1, 1), padding="valid", win_eps= 100)

    kernel_size = 91
    loss_localNCC = loss_F.LocalNCC_new(device = device, kernel_size =(kernel_size, kernel_size), stride=(2, 2), padding="valid", win_eps= 0.98)

    if args.training_mode == 'finetune':
        # overwrite the learning rate for finetune
        lr = 5e-6
        print('Learning rate is overwritten to be {}'.format(lr))
    else:
        lr = args.learning_rate
        print('Learning rate = {}'.format(lr))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    
    # num_epochs=25
    lowest_loss = 2000
    best_ep = 0
    tv_hist = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        # global current_epoch
        # current_epoch = epoch + 1

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print('Network is in {}...'.format(phase))
            
            running_loss = 0.0
            running_localNCC = 0.0
            running_dof = 0.0

            if phase == 'train':
                # scheduler.step()
                model.train()
                nth_batch = 0
                for i, batch in enumerate(training_dataset_frame):
                    """create the batch volume tensor. This way could significantly reduce the computation when preprocessing"""
                    volume_size = training_dataset_volume[0]['volume_name'].shape
                    actual_batch_size = len(batch["volume_ID"])
                    # vol_tensor = torch.zeros(actual_batch_size, volume_size[0], volume_size[1], volume_size[2], volume_size[3])

                    """converting from N*C*W*H*D to N*C*D*H*W"""
                    vol_tensor = torch.zeros(actual_batch_size, volume_size[0], volume_size[3], volume_size[2], volume_size[1])
                    for i, volume_id in enumerate(batch["volume_ID"]):
                        # print("volume_id: ", volume_id)
                        vol_tensor[i, :, :, :, :] = torch.permute(training_dataset_volume[volume_id]['volume_name'].type(torch.FloatTensor), (0, 3, 2, 1))   
                    frame_tensor = torch.permute(batch["frame_name"].type(torch.FloatTensor), (0, 1, 4, 3, 2))
                    mat_tensor = batch["tfm_gt_diff_mat"].type(torch.FloatTensor)
                    dof_tensor = batch["tfm_gt_diff_dof"].type(torch.FloatTensor)
                    
                    # print('vol_tensor {}'.format(vol_tensor.shape))
                    # print('frame_tensor {}'.format(frame_tensor.shape))
                    # print('mat_tensor {}'.format(mat_tensor.shape))
                    # print('dof_tensor {}'.format(dof_tensor.shape))
                    
                    # sys.exit()

                    vol_tensor = vol_tensor.to(device)
                    frame_tensor = frame_tensor.to(device)
                    mat_tensor = mat_tensor.to(device)
                    dof_tensor = dof_tensor.to(device)
                    mat_tensor.require_grad = True
                    dof_tensor.require_grad = True

                    frame_tensor_gt = torch.zeros((frame_tensor.shape))
                    for i, frame_flip_flag in enumerate(batch["frame_flip_flag"]):
                        if frame_flip_flag == "True":
                            frame_tensor_gt[i, :,:, :, :] = torch.flip(frame_tensor[i, :, :,:, :], [3])
                        else:
                            frame_tensor_gt[i, :, :, :,:] = frame_tensor[i,:, :, :, :]
                    frame_tensor_gt = frame_tensor_gt.type(torch.FloatTensor).to(device)
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        
                        optimizer.zero_grad()

                        vol_resampled, dof_estimated = model(vol=vol_tensor, frame=frame_tensor_gt, initial_transform = batch['tfm_RegS2V_initial_mat'] ,device=device, training=True) # shape batch_size*6
                        # print('vol_resampled {}'.format(vol_resampled.shape))
                        # print('dof_estimated {}'.format(dof_estimated))
                        

                        # """save the vol_resampled""" 
                        # vol_resampled_= torch.permute(vol_resampled.data, (0, 1, 4, 3, 2))
                        # vol_resampled_squeezed = vol_resampled_[0, :, :, :, :].squeeze()
                        # print('vol_resampled_squeezed shape {}'.format(vol_resampled_squeezed.shape))
                        # writer = ITKWriter(output_dtype=np.float32, affine_lps_to_ras= False)
                        # writer.set_data_array(vol_resampled_squeezed, channel_dim=None)
                        # writer.set_metadata(training_dataset_volume[volume_id]['volume_name'].meta, resample=False)
                        # output_filename = r"E:\PROGRAM\Project_PhD\Registration\Deepcode\test\Pre-Ablation_01_test.mha"
                        # writer.write(output_filename)

                        # sys.exit()
                        

                        """rotation loss (deg)"""
                        rotation_loss = loss_mse(dof_estimated[:, 3:], dof_tensor[:, 3:])
                        """translation loss (mm)"""
                        translation_loss = loss_mse(dof_estimated[:, :3], dof_tensor[:, :3])

                        """image intensity-based loss (localNCC)"""
                        frame_estimated = vol_resampled[:,:, int(volume_size[3]*0.5), :, :].to(device)
                        # print("frame_estimated shape: ", frame_estimated.shape)
                        

                        frame_tensor_gt_4d = frame_tensor_gt.squeeze(2)
                        # # print("frame_tensor shape: ", frame_tensor.shape)
                        # frame_tensor_gt = torch.zeros((frame_tensor_4d.shape))
                        # for i, frame_flip_flag in enumerate(batch["frame_flip_flag"]):
                        #     if frame_flip_flag == "True":
                        #         frame_tensor_gt[i, :, :, :] = torch.flip(frame_tensor_4d[i, :, :, :], [2])
                        #     else:
                        #         frame_tensor_gt[i, :, :, :] = frame_tensor_4d[i, :, :, :]

                        # frame_tensor_gt = frame_tensor_gt.type(torch.FloatTensor).to(device)
                        # print("frame_tensor_gt shape: ", frame_tensor_gt.shape)
                        ##############################################################
                        # """visualize image (tested)"""
                        # frame_gt_np = torch.Tensor.numpy(frame_tensor.detach().cpu())
                        # frame_est_np = torch.Tensor.numpy(frame_estimated.detach().cpu())
                        # plt.figure("visualize", (12,8))
                        # plt.subplot(1,2,1)
                        # plt.title("original image")
                        # plt.imshow(frame_gt_np[0,0,:,:], cmap="gray")

                        # plt.subplot(1,2,2)
                        # plt.title("resampled image")
                        # plt.imshow(frame_est_np[0,0,:,:], cmap="gray")
                        # plt.show()

                        # sys.exit()
                        
                        image_localNCC_loss, ROI = loss_localNCC(frame_estimated, frame_tensor_gt_4d)
                        # print("image_localNCC_loss: ", image_localNCC_loss)
                        # print("image_localNCC_loss device: ", image_localNCC_loss.device)
                        # coefficients for loss functinos
                        alpha = 1.0
                        beta = 1.0
                        gamma = 5.0
                        loss_combined = alpha*rotation_loss + beta*translation_loss + gamma*image_localNCC_loss
                        # loss_combined = image_localNCC_loss
                        # print("loss_combined is leaf_variable (guess False): ", loss_combined.is_leaf)
                        # print("loss_combined is required_grad (guess True): ", loss_combined.requires_grad)
                        # print("loss_combined device (guess cuda): ", loss_combined.device)
                        
                        # backward + optimize only if in training phase    
                        loss_combined.backward()
                        optimizer.step()
                        
                        # print("loss_combined: ", loss_combined)
                            
                        
                    # sys.exit()
                    running_loss += loss_combined * actual_batch_size
                    running_localNCC += image_localNCC_loss*actual_batch_size
                    running_dof += (rotation_loss + translation_loss)*actual_batch_size
                    nth_batch += 1

                    torch.cuda.empty_cache()
                    cur_lr = float(scheduler.get_last_lr()[0])
                    # print(scheduler.get_last_lr())
                    print('{}/{}: Train-BATCH (lr = {:.6f}): {:.4f}(loss_combined), {:.4f}(image_localNCC_loss), {:.4f}(loss_dof)'.format(nth_batch, math.ceil(num_cases[phase]/batch_size), cur_lr, loss_combined, image_localNCC_loss, rotation_loss+translation_loss))
                
                scheduler.step()
                # sys.exit()
                epoch_loss = running_loss / num_cases[phase]
                epoch_running_localNCC = running_localNCC/num_cases[phase]
                epoch_running_dof = running_dof/num_cases[phase]
                tv_hist[phase].append([float(epoch_loss), float(epoch_running_localNCC), float(epoch_running_dof)])
                # print('tv_hist\n{}'.format(tv_hist))
                
            else:
                model.eval()
                nth_batch = 0
                for batch in validation_dataset_frame:
                    """create the batch volume tensor. This way could significantly reduce the computation when preprocessing"""
                    volume_size = validation_dateset_volume[0]['volume_name'].shape
                    actual_batch_size = len(batch["volume_ID"])
                    # vol_tensor = torch.zeros(actual_batch_size, volume_size[0], volume_size[1], volume_size[2], volume_size[3])

                    """converting from N*C*W*H*D to N*C*D*H*W"""
                    vol_tensor = torch.zeros(actual_batch_size, volume_size[0], volume_size[3], volume_size[2], volume_size[1])
                    for i, volume_id in enumerate(batch["volume_ID"]):
                        # print("volume_id: ", volume_id)
                        vol_tensor[i, :, :, :, :] = torch.permute(validation_dateset_volume[volume_id]['volume_name'].type(torch.FloatTensor), (0, 3, 2, 1))   
                    frame_tensor = torch.permute(batch["frame_name"].type(torch.FloatTensor), (0, 1, 4, 3, 2))
                    mat_tensor = batch["tfm_gt_diff_mat"].type(torch.FloatTensor)
                    dof_tensor = batch["tfm_gt_diff_dof"].type(torch.FloatTensor)
                    

                    vol_tensor = vol_tensor.to(device)
                    frame_tensor = frame_tensor.to(device)
                    mat_tensor = mat_tensor.to(device)
                    dof_tensor = dof_tensor.to(device)

                    frame_tensor_gt = torch.zeros((frame_tensor.shape))
                    for i, frame_flip_flag in enumerate(batch["frame_flip_flag"]):
                        if frame_flip_flag == "True":
                            frame_tensor_gt[i, :,:, :, :] = torch.flip(frame_tensor[i, :, :,:, :], [3])
                        else:
                            frame_tensor_gt[i, :, :, :,:] = frame_tensor[i,:, :, :, :]
                    frame_tensor_gt = frame_tensor_gt.type(torch.FloatTensor).to(device)

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        
                        
                        vol_resampled, dof_estimated = model(vol=vol_tensor, frame=frame_tensor_gt, initial_transform = batch['tfm_RegS2V_initial_mat'] ,device=device, training=False) # shape batch_size*6
                        
                        # """save the vol_resampled""" 
                        # vol_resampled_= torch.permute(vol_resampled.data, (0, 1, 4, 3, 2))
                        # vol_resampled_squeezed = vol_resampled_[0, :, :, :, :].squeeze()
                        # print('vol_resampled_squeezed shape {}'.format(vol_resampled_squeezed.shape))
                        # writer = ITKWriter(output_dtype=np.float32, affine_lps_to_ras= False)
                        # writer.set_data_array(vol_resampled_squeezed, channel_dim=None)
                        # writer.set_metadata(training_dataset_volume[volume_id]['volume_name'].meta, resample=False)
                        # output_filename = r"E:\PROGRAM\Project_PhD\Registration\Deepcode\test\Pre-Ablation_01_test.mha"
                        # writer.write(output_filename)

                        # sys.exit()

                        """rotation loss (deg)"""
                        rotation_loss = loss_mse(dof_estimated[:, 3:], dof_tensor[:, 3:])
                        """translation loss (mm)"""
                        translation_loss = loss_mse(dof_estimated[:, :3], dof_tensor[:, :3])

                        """image intensity-based loss (localNCC)"""
                        frame_estimated = vol_resampled[:,:, int(volume_size[3]*0.5), :, :].to(device)
                       

                        frame_tensor_gt_4d = frame_tensor_gt.squeeze(2)
                        # # print("frame_tensor shape: ", frame_tensor.shape)
                        # frame_tensor_gt = torch.zeros((frame_tensor_4d.shape))
                        # for i, frame_flip_flag in enumerate(batch["frame_flip_flag"]):
                        #     if frame_flip_flag == "True":
                        #         frame_tensor_gt[i, :, :, :] = torch.flip(frame_tensor_4d[i, :, :, :], [2])
                        # frame_tensor_gt = frame_tensor_gt.type(torch.FloatTensor).to(device)
                 
                        ##############################################################
                        # """visualize image (tested)"""
                        # frame_gt_np = torch.Tensor.numpy(frame_tensor.detach().cpu())
                        # frame_est_np = torch.Tensor.numpy(frame_estimated.detach().cpu())
                        # plt.figure("visualize", (12,8))
                        # plt.subplot(1,2,1)
                        # plt.title("original image")
                        # plt.imshow(frame_gt_np[0,0,:,:], cmap="gray")

                        # plt.subplot(1,2,2)
                        # plt.title("resampled image")
                        # plt.imshow(frame_est_np[0,0,:,:], cmap="gray")
                        # plt.show()

                        # sys.exit()
                        
                        image_localNCC_loss, ROI = loss_localNCC(frame_estimated, frame_tensor_gt_4d)
                        
                        # coefficients for loss functinos
                        alpha = 1.0
                        beta = 1.0
                        gamma = 5.0
                        # loss_combined = alpha*rotation_loss + beta*translation_loss + gamma*image_localNCC_loss
                        loss_combined = image_localNCC_loss
                        # print("loss_combined is leaf_variable (guess False): ", loss_combined.is_leaf)
                        # print("loss_combined is required_grad (guess True): ", loss_combined.requires_grad)
                        # print("loss_combined device (guess cuda): ", loss_combined.device)
                        
                        # print("loss_combined: ", loss_combined)
                        
                        
                    # sys.exit()
                    running_loss += loss_combined * actual_batch_size
                    running_localNCC += image_localNCC_loss*actual_batch_size
                    running_dof += (rotation_loss + translation_loss)*actual_batch_size
                    nth_batch += 1 
                    print('{}/{}: Validation-BATCH: {:.4f}(loss_combined), {:.4f}(image_localNCC_loss), {:.4f}(loss_dof)'.format(nth_batch, math.ceil(num_cases[phase]/batch_size), loss_combined, image_localNCC_loss, rotation_loss+translation_loss))
                 
                
                # sys.exit()
                epoch_loss = running_loss / num_cases[phase]
                epoch_running_localNCC = running_localNCC/num_cases[phase]
                epoch_running_dof = running_dof/num_cases[phase]
                tv_hist[phase].append([float(epoch_loss), float(epoch_running_localNCC), float(epoch_running_dof)])
                # print('tv_hist\n{}'.format(tv_hist))
            
                # deep copy the model
                if epoch_loss < lowest_loss:
                    lowest_loss = epoch_loss
                    best_ep = epoch
                    print('**** best model updated with loss={:.4f} ****'.format(lowest_loss))
                if epoch%5 == 0 and epoch != 0:
                    fn_save = path.join(output_dir, 'RegS2V_best_{}_{}.pth'.format(net, epoch))
                    torch.save(model.state_dict(), fn_save)    
        # sys.exit()    
        # update_info(best_epoch=best_ep+1, current_epoch=epoch+1, lowest_val_TRE=lowest_loss, loss_combined = tv_hist['train'][-1][0], loss_image = tv_hist['train'][-1][1], loss_dof = tv_hist['train'][-1][2])
        update_info(best_epoch=best_ep+1, current_epoch=epoch+1, lowest_val_TRE=lowest_loss, tv_hist=tv_hist)
        print("=========================================================================================================================================")
        print('{}/{}: Train: {:.4f}(loss_combined), {:.4f}(loss_localNCC), {:.4f}(loss_dof), Validation: {:.4f}(loss_combined), {:.4f}(loss_localNCC), {:.4f}(loss_dof)'.format(
            epoch + 1, num_epochs,
            tv_hist['train'][-1][0],tv_hist['train'][-1][1], tv_hist['train'][-1][2],
            tv_hist['val'][-1][0], tv_hist['val'][-1][1], tv_hist['val'][-1][2]))
        print("=========================================================================================================================================")
        # sys.exit()
        
    time_elapsed = time.time() - since
    print('*' * 10 + 'Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return tv_hist

def train_model_initialized(model, training_dataset_frame, training_dataset_volume, validation_dataset_frame, validation_dateset_volume, num_cases):
    since = time.time()
    loss_mse = nn.MSELoss()
    # loss_localNCC = LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size= 13, kernel_type="rectangular", reduction="mean")
    # loss_localNCC = loss_F.LocalNCC(device = device, kernel_size =(71, 71), stride=(1, 1), padding="valid", win_eps= 100)
    
    # setting 1: larger image size [400, 320, 240]
    # kernel_size = 91

    # setting 2: volume size [200, 160, 120]
    kernel_size = 51
    loss_localNCC = loss_F.LocalNCC_new(device = device, kernel_size =(kernel_size, kernel_size), stride=(2, 2), padding="valid", win_eps= 0.98)

    if args.training_mode == 'finetune':
        # overwrite the learning rate for finetune
        lr = 5e-6
        print('Learning rate is overwritten to be {}'.format(lr))
    else:
        lr = args.learning_rate
        # print('Learning rate = {}'.format(lr))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.8)

    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-4,step_size_up=200)
    # scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=20)
    # now_str = '150nonorm'
    
    # output_dir = r"E:\PROGRAM\Project_PhD\Registration\Deepcode\FVR-Net\outputs\models"
    # fn_save = path.join(output_dir, 'RegS2V_best_{}_{}.pth'.format(net, now_str))
    
    # num_epochs=25
    lowest_loss = 2000
    best_ep = 0
    tv_hist = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        # global current_epoch
        # current_epoch = epoch + 1

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print('Network is in {}...'.format(phase))
            
            running_loss = 0.0
            running_localNCC = 0.0
            running_dof = 0.0

            if phase == 'train':
                # scheduler.step()
                model.train()
                nth_batch = 0
                for i, batch in enumerate(training_dataset_frame):
                    """create the batch volume tensor. This way could significantly reduce the computation when preprocessing"""
                    volume_size = training_dataset_volume[0]['volume_name'].shape
                    actual_batch_size = len(batch["volume_ID"])
                    # vol_tensor = torch.zeros(actual_batch_size, volume_size[0], volume_size[1], volume_size[2], volume_size[3])

                    """converting from N*C*W*H*D to N*C*D*H*W"""
                    vol_tensor = torch.zeros(actual_batch_size, volume_size[0], volume_size[3], volume_size[2], volume_size[1])
                    for i, volume_id in enumerate(batch["volume_ID"]):
                        # print("volume_id: ", volume_id)
                        vol_tensor[i, :, :, :, :] = torch.permute(training_dataset_volume[volume_id]['volume_name'].type(torch.FloatTensor), (0, 3, 2, 1))   
                    frame_tensor = torch.permute(batch["frame_name"].type(torch.FloatTensor), (0, 1, 4, 3, 2))
                    mat_tensor = batch["tfm_gt_diff_mat"].type(torch.FloatTensor)
                    dof_tensor = batch["tfm_gt_diff_dof"].type(torch.FloatTensor)
                    dof_tensor = dof_tensor.squeeze(1)
                    # print('vol_tensor {}'.format(vol_tensor.shape))
                    # print('frame_tensor {}'.format(frame_tensor.shape))
                    # print('mat_tensor {}'.format(mat_tensor.shape))
                    # print('dof_tensor {}'.format(dof_tensor))
                    # print('dof_tensor shape {}'.format(dof_tensor.shape))
                    # sys.exit()

                    vol_tensor = vol_tensor.to(device)
                    # frame_tensor = frame_tensor.to(device)
                    # mat_tensor = mat_tensor.to(device)
                    dof_tensor = dof_tensor.to(device)
                    # mat_tensor.require_grad = True
                    dof_tensor.require_grad = True

                    """normalize the flip of frame image"""

                    frame_tensor_gt = torch.zeros((frame_tensor.shape))
                    for i, frame_flip_flag in enumerate(batch["frame_flip_flag"]):
                        if frame_flip_flag == "True":
                            frame_tensor_gt[i, :,:, :, :] = torch.flip(frame_tensor[i, :, :,:, :], [3])
                        else:
                            frame_tensor_gt[i, :, :, :,:] = frame_tensor[i,:, :, :, :]
                    frame_tensor_gt = frame_tensor_gt.type(torch.FloatTensor).to(device)
                    frame_tensor_gt_4d = frame_tensor_gt.squeeze(2)
                    # print("frame_tensor_gt",frame_tensor_gt.shape)
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        
                        optimizer.zero_grad()

                        """initialized volume"""
                        affine_transform_initial = batch['tfm_RegS2V_initial_mat'].type(torch.FloatTensor).to(device)

                        grid_affine = F.affine_grid(theta= affine_transform_initial[:, 0:3, :], size = vol_tensor.shape, align_corners=True)
                        vol_initialized = F.grid_sample(vol_tensor, grid_affine, align_corners=True)
                        vol_initialized = vol_initialized.to(device)

                        # sys.exit()
                        vol_resampled, dof_estimated = model(vol=vol_initialized, frame=frame_tensor_gt, initial_transform = affine_transform_initial, vol_original=vol_tensor, device=device) # shape batch_size*6
                        
                        # print("dof_estimated: ", dof_estimated)
                        # print("dof_tensor: ", dof_tensor)
                        """rotation loss (deg)"""
                        rotation_loss = loss_mse(dof_estimated[:, 3:], dof_tensor[:, 3:])
                        # rotation_loss = -100
                        """translation loss (mm)"""
                        translation_loss = loss_mse(dof_estimated[:, :3], dof_tensor[:, :3])
                        # translation_loss = -100
                        """image intensity-based loss (localNCC)"""
                        frame_estimated = vol_resampled[:,:, int(volume_size[3]*0.5), :, :].to(device)
                        
                        # frame_est_np = torch.Tensor.numpy(vol_resampled[:,:, int(volume_size[3]*0.5), :, :].detach().cpu())
                        # frame_tensor_gt_4d =frame_tensor_gt_4d.detach().cpu()
                        # print("frame_estimated shape: ", frame_estimated.shape)
                        
                        """visualize the initialized images"""
                        # sampled_frame_est = vol_initialized[:,:, int(volume_size[3]*0.5), :, :]
                        # sampled_frame_est_np = torch.Tensor.numpy(sampled_frame_est.detach().cpu())

                        # affine_transform_gt = batch['tfm_RegS2V_gt_mat'].type(torch.FloatTensor).to(device)
                        # grid_affine_gt = F.affine_grid(theta= affine_transform_gt[:, 0:3, :], size = vol_tensor.shape, align_corners=True)
                        # vol_gt = F.grid_sample(vol_tensor, grid_affine_gt, align_corners=True)
                        # sampled_frame_gt = vol_gt[:,:, int(volume_size[3]*0.5), :, :]
                        # sampled_frame_gt_np = torch.Tensor.numpy(sampled_frame_gt.detach().cpu())

                        
                        # frame_gt_np = torch.Tensor.numpy(frame_tensor_gt_4d.detach().cpu())

                        # print("frame_gt_np",frame_gt_np.size)
                        # fig = plt.figure(figsize=(16, 9))
                        # ax1 = fig.add_subplot(141)
                        # ax2 = fig.add_subplot(142)
                        # ax3 = fig.add_subplot(143)
                        # ax4 = fig.add_subplot(144)
                        # ax1.imshow(frame_gt_np[0, 0, :, :], cmap = "gray")
                        # ax1.set_title("US frame (target)")
                        # ax2.imshow(sampled_frame_est_np[0, 0, :, :], cmap = "gray")
                        # ax2.set_title("Resampled image (initial)")
                        # ax3.imshow(sampled_frame_gt_np[0, 0, :, :], cmap = "gray")
                        # ax3.set_title("Resampled image (gt)")
                        # ax4.imshow(frame_est_np[0, 0, :, :], cmap = "gray")
                        # ax4.set_title("estimated image (gt)")
                        # plt.show()

                        # fig = plt.figure(figsize=(16, 9))
                        # ax1 = fig.add_subplot(141)
                        # ax2 = fig.add_subplot(142)
                        # ax3 = fig.add_subplot(143)
                        # ax4 = fig.add_subplot(144)
                        # ax1.imshow(frame_gt_np[1, 0, :, :], cmap = "gray")
                        # ax1.set_title("US frame (target)")
                        # ax2.imshow(sampled_frame_est_np[1, 0, :, :], cmap = "gray")
                        # ax2.set_title("Resampled image (initial)")
                        # ax3.imshow(sampled_frame_gt_np[1, 0, :, :], cmap = "gray")
                        # ax3.set_title("Resampled image (gt)")
                        # ax4.imshow(frame_est_np[1, 0, :, :], cmap = "gray")
                        # ax4.set_title("estimated image (gt)")
                        # plt.show()

                        # fig = plt.figure(figsize=(16, 9))
                        # ax1 = fig.add_subplot(141)
                        # ax2 = fig.add_subplot(142)
                        # ax3 = fig.add_subplot(143)
                        # ax4 = fig.add_subplot(144)
                        # ax1.imshow(frame_gt_np[2, 0, :, :], cmap = "gray")
                        # ax1.set_title("US frame (target)")
                        # ax2.imshow(sampled_frame_est_np[2, 0, :, :], cmap = "gray")
                        # ax2.set_title("Resampled image (initial)")
                        # ax3.imshow(sampled_frame_gt_np[2, 0, :, :], cmap = "gray")
                        # ax3.set_title("Resampled image (gt)")
                        # ax4.imshow(frame_est_np[2, 0, :, :], cmap = "gray")
                        # ax4.set_title("estimated image (gt)")
                        # plt.show()

                        # fig = plt.figure(figsize=(16, 9))
                        # ax1 = fig.add_subplot(141)
                        # ax2 = fig.add_subplot(142)
                        # ax3 = fig.add_subplot(143)
                        # ax4 = fig.add_subplot(144)
                        # ax1.imshow(frame_gt_np[3, 0, :, :], cmap = "gray")
                        # ax1.set_title("US frame (target)")
                        # ax2.imshow(sampled_frame_est_np[3, 0, :, :], cmap = "gray")
                        # ax2.set_title("Resampled image (initial)")
                        # ax3.imshow(sampled_frame_gt_np[3, 0, :, :], cmap = "gray")
                        # ax3.set_title("Resampled image (gt)")
                        # ax4.imshow(frame_est_np[3, 0, :, :], cmap = "gray")
                        # ax4.set_title("estimated image (gt)")
                        # plt.show()


                        # """visualize image (tested)"""
                        # # frame_gt_np = torch.Tensor.numpy(frame_tensor_gt_4d.detach().cpu())
                        # frame_est_np = torch.Tensor.numpy(frame_estimated.detach().cpu())
                        # plt.figure("visualize", (12,8))
                        # plt.subplot(1,2,1)
                        # plt.title("original image")
                        # plt.imshow(frame_gt_np[0,0,:,:], cmap="gray")

                        # plt.subplot(1,2,2)
                        # plt.title("resampled image")
                        # plt.imshow(frame_est_np[0,0,:,:], cmap="gray")
                        # plt.show()

                        # sys.exit()
                        
                        image_localNCC_loss, ROI = loss_localNCC(frame_estimated, frame_tensor_gt_4d)
                        # print("image_localNCC_loss: ", image_localNCC_loss)
                        # print("image_localNCC_loss device: ", image_localNCC_loss.device)
                        # coefficients for loss functinos
                        
                        alpha = 100.0
                        beta = 100.0
                        gamma = 1.0
                        loss_combined = alpha*rotation_loss + beta*translation_loss + gamma*image_localNCC_loss
                        # image_localNCC_loss = 0
                        # loss_combined = image_localNCC_loss
                        # loss_combined = rotation_loss + translation_loss
                        # loss_combined = alpha*rotation_loss + beta*translation_loss
                        # print("loss_combined is leaf_variable (guess False): ", loss_combined.is_leaf)
                        # print("loss_combined is required_grad (guess True): ", loss_combined.requires_grad)
                        # print("loss_combined device (guess cuda): ", loss_combined.device)
                        
                        # backward + optimize only if in training phase    
                        loss_combined.backward()
                        optimizer.step()
                        
                        # print("loss_combined: ", loss_combined)
                            
                        
                    # sys.exit()
                    running_loss += loss_combined * actual_batch_size
                    running_localNCC += image_localNCC_loss*actual_batch_size
                    running_dof += (rotation_loss + translation_loss)*actual_batch_size
                    nth_batch += 1
                    torch.cuda.empty_cache()
                    cur_lr = float(scheduler.get_last_lr()[0])
                    # print(scheduler.get_last_lr())
                    print('{}/{}: Train (lr = {:.7f}): {:.4f}(loss_combined), {:.4f}(loss_NCC), {:.6f}(loss_trans), {:.4f}(loss_rot)'.format(nth_batch, math.ceil(num_cases[phase]/batch_size), cur_lr, loss_combined, image_localNCC_loss, translation_loss, rotation_loss))
                    print("(dof-est){}".format(torch.Tensor.numpy(dof_estimated[1,:].detach().cpu())))
                    print("(dof-gt) {}".format(torch.Tensor.numpy(dof_tensor[1,:].detach().cpu())))
                    print("=========================================================================================================================================")
                # sys.exit()
                scheduler.step()
                # sys.exit()
                epoch_loss = running_loss / num_cases[phase]
                epoch_running_localNCC = running_localNCC/num_cases[phase]
                epoch_running_dof = running_dof/num_cases[phase]
                tv_hist[phase].append([float(epoch_loss), float(epoch_running_localNCC), float(epoch_running_dof)])
                # print('tv_hist\n{}'.format(tv_hist))
                
            else:
                model.eval()
                nth_batch = 0
                for batch in validation_dataset_frame:
                    """create the batch volume tensor. This way could significantly reduce the computation when preprocessing"""
                    volume_size = validation_dateset_volume[0]['volume_name'].shape
                    actual_batch_size = len(batch["volume_ID"])
                    # vol_tensor = torch.zeros(actual_batch_size, volume_size[0], volume_size[1], volume_size[2], volume_size[3])

                    """converting from N*C*W*H*D to N*C*D*H*W"""
                    vol_tensor = torch.zeros(actual_batch_size, volume_size[0], volume_size[3], volume_size[2], volume_size[1])
                    for i, volume_id in enumerate(batch["volume_ID"]):
                        # print("volume_id: ", volume_id)
                        vol_tensor[i, :, :, :, :] = torch.permute(validation_dateset_volume[volume_id]['volume_name'].type(torch.FloatTensor), (0, 3, 2, 1))   
                    frame_tensor = torch.permute(batch["frame_name"].type(torch.FloatTensor), (0, 1, 4, 3, 2))
                    # mat_tensor = batch["tfm_gt_diff_mat"].type(torch.FloatTensor)
                    dof_tensor = batch["tfm_gt_diff_dof"].type(torch.FloatTensor)
                    dof_tensor = dof_tensor.squeeze(1)

                    vol_tensor = vol_tensor.to(device)
                    frame_tensor = frame_tensor.to(device)
                    # mat_tensor = mat_tensor.to(device)
                    dof_tensor = dof_tensor.to(device)

                    frame_tensor_gt = torch.zeros((frame_tensor.shape))
                    for i, frame_flip_flag in enumerate(batch["frame_flip_flag"]):
                        if frame_flip_flag == "True":
                            frame_tensor_gt[i, :,:, :, :] = torch.flip(frame_tensor[i, :, :,:, :], [3])
                        else:
                            frame_tensor_gt[i, :, :, :,:] = frame_tensor[i,:, :, :, :]
                    frame_tensor_gt = frame_tensor_gt.type(torch.FloatTensor).to(device)

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        
                        """initialized volume"""
                        affine_transform_initial = batch['tfm_RegS2V_initial_mat'].type(torch.FloatTensor).to(device)
                        # affine_transform_initial = batch['tfm_RegS2V_initial_mat'].type(torch.FloatTensor)
                        grid_affine = F.affine_grid(theta= affine_transform_initial[:, 0:3, :], size = vol_tensor.shape, align_corners=True)
                        vol_initialized = F.grid_sample(vol_tensor, grid_affine, align_corners=True)
                        # vol_initialized = vol_initialized.to(device)
                        vol_resampled, dof_estimated = model(vol=vol_initialized, frame=frame_tensor_gt, initial_transform = affine_transform_initial, vol_original = vol_tensor ,device=device) # shape batch_size*6
                        
                        # """save the vol_resampled""" 
                        # vol_resampled_= torch.permute(vol_resampled.data, (0, 1, 4, 3, 2))
                        # vol_resampled_squeezed = vol_resampled_[0, :, :, :, :].squeeze()
                        # print('vol_resampled_squeezed shape {}'.format(vol_resampled_squeezed.shape))
                        # writer = ITKWriter(output_dtype=np.float32, affine_lps_to_ras= False)
                        # writer.set_data_array(vol_resampled_squeezed, channel_dim=None)
                        # writer.set_metadata(training_dataset_volume[volume_id]['volume_name'].meta, resample=False)
                        # output_filename = r"E:\PROGRAM\Project_PhD\Registration\Deepcode\test\Pre-Ablation_01_test.mha"
                        # writer.write(output_filename)

                        vol_size = [vol_initialized.shape[4], vol_initialized.shape[3], vol_initialized.shape[2]]
                        dof_estimated_ITK = transform_conversion_dof_normalized_to_ITK(dof_normalized=dof_estimated, vol_size = vol_size, device=device)
                        dof_gt_ITK = transform_conversion_dof_normalized_to_ITK(dof_normalized=dof_tensor, vol_size = vol_size, device=device)
                        # print("dof_estimated_ITK: ", dof_estimated_ITK)
                        # print("dof_gt_ITK: ", dof_gt_ITK)
                        # sys.exit()

                        """rotation loss (deg)"""
                        rotation_loss = loss_mse(dof_estimated[:, 3:], dof_tensor[:, 3:])
                        rotation_loss_ITK = loss_mse(dof_estimated_ITK[:, 3:], dof_gt_ITK[:, 3:])
                        """translation loss (mm)"""
                        translation_loss = loss_mse(dof_estimated[:, :3], dof_tensor[:, :3])
                        translation_loss_ITK = loss_mse(dof_estimated_ITK[:, :3], dof_gt_ITK[:, :3])

                        """image intensity-based loss (localNCC)"""
                        frame_estimated = vol_resampled[:,:, int(volume_size[3]*0.5), :, :].to(device)
                       

                        frame_tensor_gt_4d = frame_tensor_gt.squeeze(2)
                        # # print("frame_tensor shape: ", frame_tensor.shape)
                        # frame_tensor_gt = torch.zeros((frame_tensor_4d.shape))
                        # for i, frame_flip_flag in enumerate(batch["frame_flip_flag"]):
                        #     if frame_flip_flag == "True":
                        #         frame_tensor_gt[i, :, :, :] = torch.flip(frame_tensor_4d[i, :, :, :], [2])
                        # frame_tensor_gt = frame_tensor_gt.type(torch.FloatTensor).to(device)
                 
                        ##############################################################
                        # """visualize image (tested)"""
                        # frame_gt_np = torch.Tensor.numpy(frame_tensor.detach().cpu())
                        # frame_est_np = torch.Tensor.numpy(frame_estimated.detach().cpu())
                        # plt.figure("visualize", (12,8))
                        # plt.subplot(1,2,1)
                        # plt.title("original image")
                        # plt.imshow(frame_gt_np[0,0,:,:], cmap="gray")

                        # plt.subplot(1,2,2)
                        # plt.title("resampled image")
                        # plt.imshow(frame_est_np[0,0,:,:], cmap="gray")
                        # plt.show()

                        # sys.exit()
                        
                        image_localNCC_loss, ROI = loss_localNCC(frame_estimated, frame_tensor_gt_4d)
                        
                        # coefficients for loss functinos
                        alpha = 100.0
                        beta = 100.0
                        gamma = 1.0
                        loss_combined = alpha*rotation_loss + beta*translation_loss + gamma*image_localNCC_loss
                        # loss_combined = image_localNCC_loss
                        # print("loss_combined is leaf_variable (guess False): ", loss_combined.is_leaf)
                        # print("loss_combined is required_grad (guess True): ", loss_combined.requires_grad)
                        # print("loss_combined device (guess cuda): ", loss_combined.device)
                        
                        # print("loss_combined: ", loss_combined)
                            
                        
                    # sys.exit()
                    running_loss += loss_combined * actual_batch_size
                    running_localNCC += image_localNCC_loss*actual_batch_size
                    running_dof += (rotation_loss + translation_loss)*actual_batch_size  
                    nth_batch += 1
                    
                    # print('{}/{}: Train-BATCH: {:.4f}(loss_combined), {:.4f}(image_localNCC_loss), {:.6f}(loss_translation),  {:.4f}(loss_rotation)'.format(nth_batch, math.ceil(num_cases[phase]/batch_size), loss_combined, image_localNCC_loss, translation_loss, rotation_loss))
                    print('{}/{}: Train (lr = {:.7f}): {:.4f}(loss_combined), {:.4f}(loss_NCC), {:.6f}(norm){:.6f}(non-norm)(loss_trans), {:.5f}(norm){:.5f}(non-norm)(loss_rot)'.format(nth_batch, math.ceil(num_cases[phase]/batch_size), cur_lr, loss_combined, image_localNCC_loss, translation_loss, translation_loss_ITK, rotation_loss, rotation_loss_ITK))
                    print("(dof-est_ITK){}".format(torch.Tensor.numpy(dof_estimated_ITK[1,:].detach().cpu())))
                    print("(dof-gt_ITK) {}".format(torch.Tensor.numpy(dof_gt_ITK[1,:].detach().cpu())))
                    print("=========================================================================================================================================")
                # sys.exit()
                epoch_loss = running_loss / num_cases[phase]
                epoch_running_localNCC = running_localNCC/num_cases[phase]
                epoch_running_dof = running_dof/num_cases[phase]
                tv_hist[phase].append([float(epoch_loss), float(epoch_running_localNCC), float(epoch_running_dof)])
                # print('tv_hist\n{}'.format(tv_hist))
            
                # deep copy the model
                if epoch_loss < lowest_loss:
                    lowest_loss = epoch_loss
                    best_ep = epoch
                    print('**** best model updated with loss={:.4f} ****'.format(lowest_loss))
                if epoch%10 == 0 and epoch != 0:
                    fn_save = path.join(output_dir, '{}_{}_super_lr6.pth'.format(trained_model_list[str(TRAINED_MODEL)], epoch))
                    torch.save(model.state_dict(), fn_save)
                
                torch.cuda.empty_cache()    
        # sys.exit()    
        # update_info(best_epoch=best_ep+1, current_epoch=epoch+1, lowest_val_TRE=lowest_loss, loss_combined = tv_hist['train'][-1][0], loss_image = tv_hist['train'][-1][1], loss_dof = tv_hist['train'][-1][2])
        update_info(best_epoch=best_ep+1, current_epoch=epoch+1, lowest_val_TRE=lowest_loss, tv_hist=tv_hist)
        print("=========================================================================================================================================")
        print('{}/{}: Train: {:.4f}(loss_combined), {:.4f}(loss_localNCC), {:.4f}(loss_dof), Validation: {:.4f}(loss_combined), {:.4f}(loss_localNCC), {:.4f}(loss_dof)'.format(
            epoch + 1, num_epochs,
            tv_hist['train'][-1][0],tv_hist['train'][-1][1], tv_hist['train'][-1][2],
            tv_hist['val'][-1][0], tv_hist['val'][-1][1], tv_hist['val'][-1][2]))
        print("=========================================================================================================================================")
        # sys.exit()
        
    time_elapsed = time.time() - since
    print('*' * 10 + 'Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return tv_hist


def train_DeepRCS2V_model(model, training_dataset_frame, training_dataset_volume, validation_dataset_frame, validation_dateset_volume, num_cases):
    since = time.time()
    loss_mse = nn.MSELoss()
    # loss_localNCC = LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size= 13, kernel_type="rectangular", reduction="mean")
    # loss_localNCC = loss_F.LocalNCC(device = device, kernel_size =(71, 71), stride=(1, 1), padding="valid", win_eps= 100)
    # setting 1: larger image size [400, 320, 240]
    # kernel_size = 91

    # setting 2: volume size [200, 160, 120]
    kernel_size = 51
    loss_localNCC = loss_F.LocalNCC_new(device = device, kernel_size =(kernel_size, kernel_size), stride=(2, 2), padding="valid", win_eps= 0.98)

    if args.training_mode == 'finetune':
        # overwrite the learning rate for finetune
        lr = 5e-6
        print('Learning rate is overwritten to be {}'.format(lr))
    else:
        lr = args.learning_rate
        # print('Learning rate = {}'.format(lr))

    trainable_params = []
    for submodel in model.stems:
        trainable_params += list(submodel.parameters())
    trainable_params += list(model.spatial_transformer.parameters())

    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    optimizer = optim.SGD(trainable_params, lr=lr, momentum=0.9)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-4,step_size_up=100)
    # scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=20)
    # now_str = '150nonorm'
    
 
    # num_epochs=25
    lowest_loss = 2000
    best_ep = 0
    tv_hist = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print('Network is in {}...'.format(phase))
            
            running_loss = 0.0
            running_localNCC = 0.0
            running_dof = 0.0

            if phase == 'train':
                # scheduler.step()
                model.train()
                nth_batch = 0
                for i, batch in enumerate(training_dataset_frame):
                    """create the batch volume tensor. This way could significantly reduce the computation when preprocessing"""
                    volume_size = training_dataset_volume[0]['volume_name'].shape
                    actual_batch_size = len(batch["volume_ID"])
                    # vol_tensor = torch.zeros(actual_batch_size, volume_size[0], volume_size[1], volume_size[2], volume_size[3])

                    """converting from N*C*W*H*D to N*C*D*H*W"""
                    vol_tensor = torch.zeros(actual_batch_size, volume_size[0], volume_size[3], volume_size[2], volume_size[1])
                    for i, volume_id in enumerate(batch["volume_ID"]):
                        # print("volume_id: ", volume_id)
                        vol_tensor[i, :, :, :, :] = torch.permute(training_dataset_volume[volume_id]['volume_name'].type(torch.FloatTensor), (0, 3, 2, 1))   
                    frame_tensor = torch.permute(batch["frame_name"].type(torch.FloatTensor), (0, 1, 4, 3, 2))
                    # mat_tensor = batch["tfm_gt_diff_mat"].type(torch.FloatTensor)
                    dof_tensor = batch["tfm_gt_diff_dof"].type(torch.FloatTensor)
                    
                    # print('vol_tensor {}'.format(vol_tensor.shape))
                    # print('frame_tensor {}'.format(frame_tensor.shape))
                    # print('mat_tensor {}'.format(mat_tensor.shape))
                    # print('dof_tensor {}'.format(dof_tensor.shape))
                    
                    # sys.exit()
                    vol_tensor = vol_tensor.to(device)
                    # frame_tensor = frame_tensor.to(device)
                    # mat_tensor = mat_tensor.to(device)
                    dof_tensor = dof_tensor.to(device)
                    # mat_tensor.require_grad = True
                    # dof_tensor.require_grad = True

                    """normalize the flip of frame image"""
                    frame_tensor_gt = torch.zeros((frame_tensor.shape))
                    for i, frame_flip_flag in enumerate(batch["frame_flip_flag"]):
                        if frame_flip_flag == "True":
                            frame_tensor_gt[i, :,:, :, :] = torch.flip(frame_tensor[i, :, :,:, :], [3])
                        else:
                            frame_tensor_gt[i, :, :, :,:] = frame_tensor[i,:, :, :, :]
                    frame_tensor_gt = frame_tensor_gt.type(torch.FloatTensor).to(device)

                    # print("frame_tensor_gt",frame_tensor_gt.shape)
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        optimizer.zero_grad()

                        """initialized volume"""
                        affine_transform_initial = batch['tfm_RegS2V_initial_mat'].type(torch.FloatTensor).to(device)
                        # # vol_initialized = vol_initialized.to(device)
                        frame_tensor_gt_4d = frame_tensor_gt.squeeze(2)
                        # sys.exit()
                        vol_resampled, theta_list, delta_mat_accumulate_list = model(vol=vol_tensor, img_fixed=frame_tensor_gt_4d, initial_transform = affine_transform_initial) # shape batch_size*6
                        
                        # cascaded_transform = calculate_cascaded_transform(theta_list, device)
                        # print('cascaded_transform: {}'.format(cascaded_transform))
                        # print("theta_list: {}".format(theta_list))
                        # print('delta_mat_accumulate_list: {}'.format(delta_mat_accumulate_list))
                        # print('delta_mat_accumulate_list[-1]: {}'.format(delta_mat_accumulate_list[-1]))
                        # sys.exit()
                        dof_estimated = tools.mat2dof_tensor(delta_mat_accumulate_list[-1]) # non-normalized
                        dof_estimated = dof_estimated.to(device)
                        """rotation loss (deg)"""
                        rotation_loss = loss_mse(dof_estimated[:, 3:], dof_tensor[:, 3:])
                        """translation loss (mm)"""
                        translation_loss = loss_mse(dof_estimated[:, :3], dof_tensor[:, :3])

                        """image intensity-based loss (localNCC)"""
                        frame_estimated = vol_resampled[:,:, int(volume_size[3]*0.5), :, :]


                        # # print("frame_estimated shape: ", frame_estimated.shape)
                        # """visualize the initialized images"""
                        # frame_gt_np = torch.Tensor.numpy(frame_tensor_gt_4d.detach().cpu())
                        # sampled_frame_est_np = torch.Tensor.numpy(frame_estimated.detach().cpu())
                        
                        # grid_affine = F.affine_grid(theta= affine_transform_initial[:, 0:3, :], size = vol_tensor.shape, align_corners=True)
                        # vol_initialized = F.grid_sample(vol_tensor, grid_affine, align_corners=True)
                        # img_moving = vol_initialized[:,:, int(volume_size[3]*0.5), :, :]
                        # sampled_frame_ini_np = torch.Tensor.numpy(img_moving.detach().cpu())

                        # affine_transform_gt = batch['tfm_RegS2V_gt_mat'].type(torch.FloatTensor).to(device)
                        # grid_affine_gt = F.affine_grid(theta= affine_transform_gt[:, 0:3, :], size = vol_tensor.shape, align_corners=True)
                        # vol_gt = F.grid_sample(vol_tensor, grid_affine_gt, align_corners=True)
                        # sampled_frame_gt = vol_gt[:,:, int(volume_size[3]*0.5), :, :]
                        # sampled_frame_gt_np = torch.Tensor.numpy(sampled_frame_gt.detach().cpu())
                        # # print("frame_gt_np",frame_gt_np.size)

                        # fig = plt.figure(figsize=(16, 9))
                        # ax1 = fig.add_subplot(141)
                        # ax2 = fig.add_subplot(142)
                        # ax3 = fig.add_subplot(143)
                        # ax4 = fig.add_subplot(144)
                        # ax1.imshow(frame_gt_np[0, 0, :, :], cmap = "gray")
                        # ax1.set_title("US frame (target)")
                        # ax2.imshow(sampled_frame_ini_np[0, 0, :, :], cmap = "gray")
                        # ax2.set_title("Resampled image (initial)")
                        # ax3.imshow(sampled_frame_est_np[0, 0, :, :], cmap = "gray")
                        # ax3.set_title("Resampled image (est)")
                        # ax4.imshow(sampled_frame_gt_np[0, 0, :, :], cmap = "gray")
                        # ax4.set_title("Resampled image (gt)")
                        # plt.show()


                        # sys.exit()
                        
                        image_localNCC_loss, ROI = loss_localNCC(frame_estimated, frame_tensor_gt_4d)
                        # print("image_localNCC_loss: ", image_localNCC_loss)
                        # print("image_localNCC_loss device: ", image_localNCC_loss.device)
                        # coefficients for loss functinos
                        
                        alpha = 1.0
                        beta = 1.0
                        gamma = 20.0
                        loss_combined = alpha*rotation_loss + beta*translation_loss + gamma*image_localNCC_loss
                        # loss_combined = image_localNCC_loss
                        # print("loss_combined is leaf_variable (guess False): ", loss_combined.is_leaf)
                        # print("loss_combined is required_grad (guess True): ", loss_combined.requires_grad)
                        # print("loss_combined device (guess cuda): ", loss_combined.device)
                        
                        # backward + optimize only if in training phase    
                        loss_combined.backward()
                        optimizer.step()
                        
                        # print("loss_combined: ", loss_combined)
                            
                        
                    # sys.exit()
                    running_loss += loss_combined * actual_batch_size
                    running_localNCC += image_localNCC_loss*actual_batch_size
                    running_dof += (rotation_loss + translation_loss)*actual_batch_size
                    nth_batch += 1
                    torch.cuda.empty_cache()
                    cur_lr = float(scheduler.get_last_lr()[0])
                    # print(scheduler.get_last_lr())
                    # print('{}/{}: Train-BATCH (lr = {:.7f}): {:.4f}(loss_combined), {:.4f}(image_localNCC_loss), {:.4f}(loss_dof)'.format(nth_batch, math.ceil(num_cases[phase]/batch_size), cur_lr, loss_combined, image_localNCC_loss, rotation_loss+translation_loss))
                    print('{}/{}: Train-BATCH (lr = {:.7f}): {:.4f}(loss_combined), {:.4f}(image_localNCC_loss), {:.6f}(loss_translation), {:.4f}(loss_rotation)'.format(nth_batch, math.ceil(num_cases[phase]/batch_size), cur_lr, loss_combined, image_localNCC_loss, translation_loss, rotation_loss))
                scheduler.step()
                # sys.exit()
                epoch_loss = running_loss / num_cases[phase]
                epoch_running_localNCC = running_localNCC/num_cases[phase]
                epoch_running_dof = running_dof/num_cases[phase]
                tv_hist[phase].append([float(epoch_loss), float(epoch_running_localNCC), float(epoch_running_dof)])
                # print('tv_hist\n{}'.format(tv_hist))
                
            else:
                model.eval()
                nth_batch = 0
                for batch in validation_dataset_frame:
                    """create the batch volume tensor. This way could significantly reduce the computation when preprocessing"""
                    volume_size = validation_dateset_volume[0]['volume_name'].shape
                    actual_batch_size = len(batch["volume_ID"])
                    # vol_tensor = torch.zeros(actual_batch_size, volume_size[0], volume_size[1], volume_size[2], volume_size[3])

                    """converting from N*C*W*H*D to N*C*D*H*W"""
                    vol_tensor = torch.zeros(actual_batch_size, volume_size[0], volume_size[3], volume_size[2], volume_size[1])
                    for i, volume_id in enumerate(batch["volume_ID"]):
                        # print("volume_id: ", volume_id)
                        vol_tensor[i, :, :, :, :] = torch.permute(validation_dateset_volume[volume_id]['volume_name'].type(torch.FloatTensor), (0, 3, 2, 1))   
                    frame_tensor = torch.permute(batch["frame_name"].type(torch.FloatTensor), (0, 1, 4, 3, 2))
                    # mat_tensor = batch["tfm_gt_diff_mat"].type(torch.FloatTensor)
                    dof_tensor = batch["tfm_gt_diff_dof"].type(torch.FloatTensor)
                    

                    vol_tensor = vol_tensor.to(device)
                    frame_tensor = frame_tensor.to(device)
                    # mat_tensor = mat_tensor.to(device)
                    dof_tensor = dof_tensor.to(device)

                    frame_tensor_gt = torch.zeros((frame_tensor.shape))
                    for i, frame_flip_flag in enumerate(batch["frame_flip_flag"]):
                        if frame_flip_flag == "True":
                            frame_tensor_gt[i, :,:, :, :] = torch.flip(frame_tensor[i, :, :,:, :], [3])
                        else:
                            frame_tensor_gt[i, :, :, :,:] = frame_tensor[i,:, :, :, :]
                    frame_tensor_gt = frame_tensor_gt.type(torch.FloatTensor).to(device)

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        
                        """initialized volume"""
                        affine_transform_initial = batch['tfm_RegS2V_initial_mat'].type(torch.FloatTensor).to(device)
                        # affine_transform_initial = batch['tfm_RegS2V_initial_mat'].type(torch.FloatTensor)
                        # grid_affine = F.affine_grid(theta= affine_transform_initial[:, 0:3, :], size = vol_tensor.shape, align_corners=True)
                        # vol_initialized = F.grid_sample(vol_tensor, grid_affine, align_corners=True)
                        # img_moving = vol_initialized[:,:, int(volume_size[3]*0.5), :, :]
                        frame_tensor_gt_4d = frame_tensor_gt.squeeze(2)
                        vol_resampled, theta_list, delta_mat_accumulate_list = model(vol=vol_tensor, img_fixed=frame_tensor_gt_4d, initial_transform = affine_transform_initial) # shape batch_size*6
                        
                        # sys.exit()
                        # cascaded_transform = calculate_cascaded_transform(theta_list, device)

                        dof_estimated = tools.mat2dof_tensor(delta_mat_accumulate_list[-1])
                        dof_estimated = dof_estimated.to(device)

                        """rotation loss (deg)"""
                        rotation_loss = loss_mse(dof_estimated[:, 3:], dof_tensor[:, 3:])
                        """translation loss (mm)"""
                        translation_loss = loss_mse(dof_estimated[:, :3], dof_tensor[:, :3])

                        """image intensity-based loss (localNCC)"""
                        frame_estimated = vol_resampled[:,:, int(volume_size[3]*0.5), :, :]

                        # frame_tensor_gt_4d = frame_tensor_gt.squeeze(2)
                        # # print("frame_tensor shape: ", frame_tensor.shape)
                        # frame_tensor_gt = torch.zeros((frame_tensor_4d.shape))
                        # for i, frame_flip_flag in enumerate(batch["frame_flip_flag"]):
                        #     if frame_flip_flag == "True":
                        #         frame_tensor_gt[i, :, :, :] = torch.flip(frame_tensor_4d[i, :, :, :], [2])
                        # frame_tensor_gt = frame_tensor_gt.type(torch.FloatTensor).to(device)
                 
                        ##############################################################
                        # """visualize image (tested)"""
                        # frame_gt_np = torch.Tensor.numpy(frame_tensor.detach().cpu())
                        # frame_est_np = torch.Tensor.numpy(frame_estimated.detach().cpu())
                        # plt.figure("visualize", (12,8))
                        # plt.subplot(1,2,1)
                        # plt.title("original image")
                        # plt.imshow(frame_gt_np[0,0,:,:], cmap="gray")

                        # plt.subplot(1,2,2)
                        # plt.title("resampled image")
                        # plt.imshow(frame_est_np[0,0,:,:], cmap="gray")
                        # plt.show()

                        # sys.exit()
                        
                        image_localNCC_loss, ROI = loss_localNCC(frame_estimated, frame_tensor_gt_4d)
                        
                        # coefficients for loss functinos
                        # alpha = 1.0
                        # beta = 2.0
                        # gamma = 10.0
                        # loss_combined = alpha*rotation_loss + beta*translation_loss + gamma*image_localNCC_loss
                        loss_combined = image_localNCC_loss
                        # print("loss_combined is leaf_variable (guess False): ", loss_combined.is_leaf)
                        # print("loss_combined is required_grad (guess True): ", loss_combined.requires_grad)
                        # print("loss_combined device (guess cuda): ", loss_combined.device)
                        
                        # print("loss_combined: ", loss_combined)
                            
                        
                    # sys.exit()
                    running_loss += loss_combined * actual_batch_size
                    running_localNCC += image_localNCC_loss*actual_batch_size
                    running_dof += (rotation_loss + translation_loss)*actual_batch_size  
                    nth_batch += 1
                    
                    # print('{}/{}: Train-BATCH: {:.4f}(loss_combined), {:.4f}(image_localNCC_loss), {:.4f}(loss_dof)'.format(nth_batch, math.ceil(num_cases[phase]/batch_size), loss_combined, image_localNCC_loss, rotation_loss+translation_loss))
                    print('{}/{}: Train-BATCH: {:.4f}(loss_combined), {:.4f}(image_localNCC_loss), {:.6f}(loss_translation),  {:.4f}(loss_rotation)'.format(nth_batch, math.ceil(num_cases[phase]/batch_size), loss_combined, image_localNCC_loss, translation_loss, rotation_loss))
                # sys.exit()
                epoch_loss = running_loss / num_cases[phase]
                epoch_running_localNCC = running_localNCC/num_cases[phase]
                epoch_running_dof = running_dof/num_cases[phase]
                tv_hist[phase].append([float(epoch_loss), float(epoch_running_localNCC), float(epoch_running_dof)])
                # print('tv_hist\n{}'.format(tv_hist))
            
                # deep copy the model
                if epoch_loss < lowest_loss:
                    lowest_loss = epoch_loss
                    best_ep = epoch
                    print('**** best model updated with loss={:.4f} ****'.format(lowest_loss))
                if epoch%5 == 0 and epoch != 0:
                    fn_save = path.join(output_dir, '{}_{}.pth'.format(trained_model_list[str(TRAINED_MODEL)], epoch))
                    torch.save(model.state_dict(), fn_save)
                
                torch.cuda.empty_cache()    
        # sys.exit()    
        # update_info(best_epoch=best_ep+1, current_epoch=epoch+1, lowest_val_TRE=lowest_loss, loss_combined = tv_hist['train'][-1][0], loss_image = tv_hist['train'][-1][1], loss_dof = tv_hist['train'][-1][2])
        update_info(best_epoch=best_ep+1, current_epoch=epoch+1, lowest_val_TRE=lowest_loss, tv_hist=tv_hist)
        print("=========================================================================================================================================")
        print('{}/{}: Train: {:.4f}(loss_combined), {:.4f}(loss_localNCC), {:.4f}(loss_dof), Validation: {:.4f}(loss_combined), {:.4f}(loss_localNCC), {:.4f}(loss_dof)'.format(
            epoch + 1, num_epochs,
            tv_hist['train'][-1][0],tv_hist['train'][-1][1], tv_hist['train'][-1][2],
            tv_hist['val'][-1][0], tv_hist['val'][-1][1], tv_hist['val'][-1][2]))
        print("=========================================================================================================================================")
        # sys.exit()
        
    time_elapsed = time.time() - since
    print('*' * 10 + 'Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return tv_hist


def train_nondeep_model(dataset_frame, dataset_volume, frame_index):
    
    """Pre-setting (same as ITK global one)"""
    lr_localNCC = 0.001
    num_iters = 100
    stop_thrld = 1e-4

    """loading volume"""
    volume = dataset_volume[dataset_frame[frame_index]["volume_ID"]]["volume_name"]
    volume = volume.unsqueeze(0).type(torch.FloatTensor)
    volume = torch.permute(volume, (0, 1, 4, 3, 2)).type(torch.FloatTensor)
    volume = volume.to(device)
    volume_size = dataset_volume[0]['volume_name'].shape
    
    loss_mse = nn.MSELoss()
    # loss_localNCC = LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size= 13, kernel_type="rectangular", reduction="mean")
    loss_localNCC = loss_F.LocalNCC(device = device, kernel_size =(71, 71), stride=(1, 1), padding="valid", win_eps= 100)
    
    starting = time.time()
    """initialize rotation and transaltion vector"""
    rtvec_est = torch.zeros((1, 6), requires_grad = True, device = device)
    # print("rtvec_est is leaf_variable (guess True): ", rtvec_est.is_leaf)
    
    rtvec_gt = dataset_frame[frame_index]["tfm_gt_diff_dof"].type(torch.FloatTensor)
    rtvec_gt = rtvec_gt.unsqueeze(0).type(torch.FloatTensor)
    rtvec_gt = rtvec_gt.to(device)
    # print("rtvec_gt shape:{}".format(rtvec_gt.shape))
    # print("rtvec_gt: {}".format(rtvec_gt))

    """loading the initial transformation from (arm tracking + transformaion of 3DUS-CT/MRI)"""
    initial_transform = dataset_frame[frame_index]["tfm_RegS2V_initial_mat"]
    affine_transform_initial = initial_transform.type(torch.FloatTensor).to(device)
    # print(affine_transform_initial.shape)
    optimizer_nondeep_model = optim.SGD([rtvec_est], lr=lr_localNCC, momentum=0.9)        
    scheduler_nondeep_model = lr_scheduler.CyclicLR(optimizer_nondeep_model, base_lr=0.001, max_lr=0.003,step_size_up=20)
    
    lossLocalNCC_output_list = torch.zeros(1, device=device)
    lossRot_output_list = torch.zeros(1, device = device)
    lossTrans_output_list = torch.zeros(1, device = device)
    
    for iter in range(num_iters):

        optimizer_nondeep_model.zero_grad()

        """loading frame"""
        frame_tensor = torch.permute(dataset_frame[frame_index]["frame_name"].type(torch.FloatTensor), (0, 3, 2, 1)).to(device)
        frame_flip_flag = dataset_frame[frame_index]["frame_flip_flag"]
        if frame_flip_flag == "True":
            frame_tensor = torch.flip(frame_tensor, [3])
        frame_tensor = frame_tensor.type(torch.FloatTensor)
        
        """get transformation"""
        correction_transform = tools.dof2mat_tensor(input_dof=rtvec_est).type(torch.FloatTensor).to(device)
        affine_transform_combined = torch.matmul(correction_transform, affine_transform_initial)
        # print(correction_transform.shape)
        # print(affine_transform_combined.shape)
        """Resampling and reslicing the volume """
        grid_affine = F.affine_grid(theta= affine_transform_combined[:, 0:3, :], size = torch.Size([1, 1, volume_size[3], volume_size[2], volume_size[1]]), align_corners=True)
        
        """get resampled volume and frame"""
        vol_resampled = F.grid_sample(volume, grid_affine, align_corners=True)
        # resampled_image_initial_np = torch.Tensor.numpy(vol_resampled)
        sampled_frame_estimated = vol_resampled[:,:, int(volume_size[3]*0.5), :, :].to(device)
        
        """rotation loss (deg)"""
        rotation_loss = loss_mse(rtvec_est[:, 3:], rtvec_gt[:, 3:])
        """translation loss (mm)"""
        translation_loss = loss_mse(rtvec_est[:, :3], rtvec_gt[:, :3])

        """image intensity-based loss (localNCC)"""
        image_localNCC_loss, ROI = loss_localNCC(sampled_frame_estimated, frame_tensor)
        
        # loss_combined = image_localNCC_loss
        
        rtvec_est.retain_grad()
        image_localNCC_loss.backward()
        optimizer_nondeep_model.step()
        scheduler_nondeep_model.step()
        # frame_gt_np = torch.Tensor.numpy(frame_tensor.detach().cpu())
        # sampled_frame_estimated_np = torch.Tensor.numpy(sampled_frame_estimated.detach().cpu())
        if iter == 0:
            frame_gt_np = torch.Tensor.numpy(frame_tensor.detach().cpu())
            sampled_frame_estimated_np = torch.Tensor.numpy(sampled_frame_estimated.detach().cpu())
            initial_sampled_frame_est_np = sampled_frame_estimated_np
            lossLocalNCC_output_list = image_localNCC_loss.reshape(1)
            lossRot_output_list = rotation_loss.reshape(1)
            lossTrans_output_list = translation_loss.reshape(1)
        else:
            lossLocalNCC_output_list =torch.cat((lossLocalNCC_output_list, image_localNCC_loss.reshape(1)))
            lossRot_output_list = torch.cat((lossRot_output_list, rotation_loss.reshape(1)))
            lossTrans_output_list = torch.cat((lossTrans_output_list, translation_loss.reshape(1)))

        if iter > 10:
            stop_flag = torch.std(lossLocalNCC_output_list[-5:]) < stop_thrld
            if stop_flag:
                sampled_frame_estimated_np = torch.Tensor.numpy(sampled_frame_estimated.detach().cpu())
                break
            
    end = time.time()
    time_elapsed = end - starting
    print('*' * 10 + 'Training complete in {:.2f}s'.format(time_elapsed) + '*' * 10)
    
    lossLocalNCC_output_list_np = torch.Tensor.numpy(lossLocalNCC_output_list.detach().cpu())
    lossRot_output_list_np = torch.Tensor.numpy(lossRot_output_list.detach().cpu())
    lossTrans_output_list_np = torch.Tensor.numpy(lossTrans_output_list.detach().cpu())

    """get registered image using ITK global NCC"""
    transform_gt = dataset_frame[frame_index]["tfm_RegS2V_gt_mat"]
    affine_transform_gt = transform_gt.type(torch.FloatTensor).to(device)
    affine_transform_gt = affine_transform_gt.unsqueeze(0)
    # print(affine_transform_gt)
    # print(affine_transform_gt.shape)
    """Resampling and reslicing the volume """
    grid_affine_gt = F.affine_grid(theta= affine_transform_gt[:, 0:3, :], size = torch.Size([1, 1, volume_size[3], volume_size[2], volume_size[1]]), align_corners=True)
    """get resampled volume and frame"""
    vol_resampled_gt = F.grid_sample(volume, grid_affine_gt, align_corners=True)
    # resampled_image_initial_np = torch.Tensor.numpy(vol_resampled)
    sampled_frame_gt = vol_resampled_gt[:,:, int(volume_size[3]*0.5), :, :]
    sampled_frame_gt_np = torch.Tensor.numpy(sampled_frame_gt.detach().cpu())
    # fig = plt.figure(figsize=(15, 9))
    # util_plot.plot_nondeep_model_comb(fig, lossLocalNCC_output_list_np, lossRot_output_list_np, lossTrans_output_list_np, initial_sampled_frame_est_np, sampled_frame_estimated_np, frame_gt_np)
    # util_plot.plot_nondeep_model_comb_with_ITK(fig, lossLocalNCC_output_list_np, lossRot_output_list_np, lossTrans_output_list_np, initial_sampled_frame_est_np, sampled_frame_estimated_np, frame_gt_np, sampled_frame_gt_np)


if __name__ == '__main__':
    
    """load dataset setting file and create the dictionary"""
    
    # project_dir = 'E:\PROGRAM\Project_PhD\Registration\DeepRegS2V'
    project_dir = os.getcwd() # Note: direct to the project folder
    data_tree_file = os.path.join(project_dir, "src/dataset_index.xml")
    if platform.system() == 'Linux':
        data_tree_file = '/'.join(data_tree_file.split('\\'))
    data_tree = ET.parse(data_tree_file)
    root = data_tree.getroot()
    # all_cases_metadata = root.find('all_cases')
    training_cases_metadata = root.find("training_cases")
    validation_cases_metadata = root.find("validation_cases")

    # alldataset_dict, volume_dict = CreateLookupTable(all_cases_metadata, save_flag=True)
    training_dataset_dict, training_volume_dict = CreateLookupTable(training_cases_metadata,project_dir= project_dir, phase= "train", save_flag=True)
    validation_dataset_dict, validation_volume_dict = CreateLookupTable(validation_cases_metadata, project_dir= project_dir, phase= "val", save_flag=True)
    
    """preprocessing the dataset(volume data, 2D US frame data and transformation data)"""
    # """preprocessing: setting 1"""
    # resample_spacing = 0.5
    # resize_scale = 1/resample_spacing
    # volume_size = [400, 320, 240]

    """preprocessing: setting 2"""
    resample_spacing = 1
    resize_scale = 1/resample_spacing
    volume_size = [200, 160, 120]

    # preprocess the 3D US volume data
    transform_3DUS = Compose(
        [
            LoadImaged(keys=["volume_name", "volume_mask_name"], reader=ITKReader(reverse_indexing=False, affine_lps_to_ras=False), image_only=False), # N*C*D*H*W
            # MaskIntensityd(keys=["image","image_mask"], mask_data= mask_image_array),
            MaskIntensityd(keys=["volume_name"], mask_key= "volume_mask_name"),
            EnsureChannelFirstd(keys = ["volume_name"]),
            Spacingd(keys=["volume_name"], pixdim=(resample_spacing, resample_spacing, resample_spacing), mode=("bilinear")),
            SpatialPadd(keys=["volume_name"], spatial_size=[volume_size[0], volume_size[1], volume_size[2]], method="symmetric", mode="constant"),
            CenterSpatialCropd(keys=["volume_name"], roi_size=[volume_size[0], volume_size[1], volume_size[2]]), # when the spacing is 0.5*0.5*0.5 (W*H*D)
            # CenterSpatialCropd(keys=["volume_name"], roi_size=[260, 300, 380]), # when the spacing is 0.5*0.5*0.5 (D*H*W)
            ScaleIntensityd(keys=["volume_name"], minv= 0.0, maxv = 1.0, dtype= np.float32),
        ]
    )
    # prepocess the 2D US and registration transformation data
    transform_2DUS = Compose(
        [
            LoadRegistrationTransformd(keys=["tfm_RegS2V"], scale=resize_scale, volume_size=volume_size),
            LoadImaged(keys=["frame_name", "frame_mask_name"], reader=ITKReader(reverse_indexing=False, affine_lps_to_ras=False), image_only=False),
            # MaskIntensityd(keys=["image","image_mask"], mask_data= mask_image_array),
            MaskIntensityd(keys=["frame_name", "frame_mask_name"], mask_key= "frame_mask_name"),
            EnsureChannelFirstd(keys = ["frame_name", "frame_mask_name"]),
            Spacingd(keys=["frame_name", "frame_mask_name"], pixdim=(resample_spacing, resample_spacing, resample_spacing), mode=("bilinear")),
            SpatialPadd(keys=["frame_name", "frame_mask_name"], spatial_size=[volume_size[0], volume_size[1], 1], method="symmetric", mode="constant"),
            CenterSpatialCropd(keys=["frame_name", "frame_mask_name"], roi_size=[volume_size[0], volume_size[1], 1]), # when the spacing is 0.5*0.5*0.5
            ScaleIntensityd(keys=["frame_name", "frame_mask_name"], minv= 0.0, maxv = 1.0, dtype= np.float32)
        ]
    )
    

    training_dataset_3DUS = CacheDataset(data=training_volume_dict, transform=transform_3DUS)
    training_dataset_2DUS = CacheDataset(data=training_dataset_dict, transform=transform_2DUS)
    
    validation_dataset_3DUS = CacheDataset(data=validation_volume_dict, transform=transform_3DUS)
    validation_dataset_2DUS = CacheDataset(data=validation_dataset_dict, transform=transform_2DUS)
    

    if DEEP_MODEL:
        training_dataloader_2DUS = DataLoader(dataset=training_dataset_2DUS, batch_size=batch_size, shuffle=True)
        validation_dataloader_2DUS = DataLoader(dataset=validation_dataset_2DUS, batch_size=batch_size, shuffle=False)
        # print("number of cases: ", len(training_dataset_2DUS))
        # sys.exit()
        num_cases = {'train': len(training_dataset_2DUS), 'val': len(validation_dataset_2DUS)}
        # Define the training model architecture
        if TRAINED_MODEL == 1:
            model = RegS2Vnet.mynet3(layers=[3, 8, 36, 3]).to(device=device)
            if RESUME_MODEL:
                # pretrained_model = path.join("/home/UWO/xshuwei/DeepRegS2V/src/outputs_1/", 'RegS2V_best_Generator_30.pth') # Yan et al.'s method pretrained model
                pretrained_model = path.join("/home/UWO/xshuwei/DeepRegS2V/src/outputs_featurefusion/", 'RegS2V_feature_fusion_Generator_10.pth') # Yan et al.'s method pretrained model
                # pretrained_model = path.join("/home/UWO/xshuwei/DeepRegS2V/src/outputs_DeepS2VFF_plus_noise/", 'DeepS2VFF_training_scratch_5_dofonly.pth') #  pretrained model
                model.load_state_dict(torch.load(pretrained_model, map_location=device))
                print("RESUME model: {}".format(pretrained_model))
            tv_hist = train_model(model=model, training_dataset_frame=training_dataloader_2DUS, training_dataset_volume = training_dataset_3DUS, validation_dataset_frame = validation_dataloader_2DUS, validation_dateset_volume = validation_dataset_3DUS, num_cases= num_cases)
        if TRAINED_MODEL == 2:
            model = RegS2Vnet.RegS2Vnet_featurefusion(layers=[3, 3, 8, 3]).to(device=device)
            if RESUME_MODEL:
                
                pretrained_model = path.join("/home/UWO/xshuwei/DeepRegS2V/src/outputs_DeepS2VFF_plus_noise/", 'DeepS2VFF_training_scratch_5_dofonly.pth') #  pretrained model
                model.load_state_dict(torch.load(pretrained_model, map_location=device))
                print("RESUME model: {}".format(pretrained_model))
            tv_hist = train_model_initialized(model=model, training_dataset_frame=training_dataloader_2DUS, training_dataset_volume = training_dataset_3DUS, validation_dataset_frame = validation_dataloader_2DUS, validation_dateset_volume = validation_dataset_3DUS, num_cases= num_cases)
        
        if TRAINED_MODEL == 3:
            model = RegS2Vnet.DeepRCS2V(num_cascades=3, device= device)
            if RESUME_MODEL:
                pretrained_model = path.join("/home/UWO/xshuwei/DeepRegS2V/src/outputs_featurefusion/", 'XXX.pth') # 
                model.load_state_dict(torch.load(pretrained_model, map_location=device))
                print("RESUME model: {}".format(pretrained_model))
            tv_hist = train_DeepRCS2V_model(model=model, training_dataset_frame=training_dataloader_2DUS, training_dataset_volume = training_dataset_3DUS, validation_dataset_frame = validation_dataloader_2DUS, validation_dateset_volume = validation_dataset_3DUS, num_cases= num_cases)
        
        if TRAINED_MODEL == 4:
            model = RegS2Vnet.RegS2Vnet_featurefusion_simplified().to(device=device)
            if RESUME_MODEL:
                pretrained_model = path.join("/home/UWO/xshuwei/DeepRegS2V/src/outputs_DeepS2VFF_simplified_unsupervised_1/", 'DeepS2VFF_simplified_490_weak_super.pth') # 
                model.load_state_dict(torch.load(pretrained_model, map_location=device))
                print("RESUME model: {}".format(pretrained_model))
            tv_hist = train_model_initialized(model=model, training_dataset_frame=training_dataloader_2DUS, training_dataset_volume = training_dataset_3DUS, validation_dataset_frame = validation_dataloader_2DUS, validation_dateset_volume = validation_dataset_3DUS, num_cases= num_cases)

        json_obj = json.dumps(tv_hist)
        f = open(os.path.join(output_dir, 'results_'+'{}.json'.format(now_str)), 'w')
        # write json object to file
        f.write(json_obj)
        # close file
        f.close()

    if NONDEEP_MODEL:
        frame_index = 6
        train_nondeep_model(training_dataset_2DUS, training_dataset_3DUS, frame_index)