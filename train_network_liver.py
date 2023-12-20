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
                    default=1e-4)

parser.add_argument('-d', '--device_no',
                    type=int,
                    choices=[0, 1, 2, 3, 4, 5, 6, 7],
                    help='GPU device number [0-7]',
                    default=0)

parser.add_argument('-e', '--epochs',
                    type=int,
                    help='number of training epochs',
                    default=5)

parser.add_argument('-n', '--network_type',
                    type=str,
                    help='choose different network architectures'
                         'the size of inputs/outputs are the same'
                         'could be original, resnext101',
                    default='mynet')


pretrain_model_str = '0213-092230'

networks3D = ['resnext50', 'resnext101', 'densenet121', 'mynet', 'mynet2', 'p3d',
              'autoencoder', 'uda']

net = 'Generator'

use_last_pretrained = False
current_epoch = 0

args = parser.parse_args()
device_no = args.device_no
device = torch.device("cuda:{}".format(device_no))

num_epochs = args.epochs
print('target training epoches is {}'.format(num_epochs))
print('start device {}'.format(device))

project_dir = os.getcwd()
output_dir = os.path.join(project_dir, "src/outputs")
print(output_dir)
now = datetime.now()
now_str = now.strftime('%m%d_%H%M%S')
print('now_str: {}'.format(now_str))
# create saving file
file = open(os.path.join(output_dir, '{}.txt'.format(now_str)), 'w')
file.close()

def data_transform(input_img, crop_size=224, resize=224, normalize=False, masked_full=False):
    """
    Crop and resize image as you wish. This function is shared through multiple scripts
    :param input_img: please input a grey-scale numpy array image
    :param crop_size: center crop size, make sure do not contain regions beyond fan-shape
    :param resize: resized size
    :param normalize: whether normalize the image
    :return: transformed image
    """
    if masked_full:
        input_img[fan_mask == 0] = 0
        masked_full_img = input_img[112:412, 59:609]
        return masked_full_img

    h, w = input_img.shape
    if crop_size > 480:
        crop_size = 480
    x_start = int((h - crop_size) / 2)
    y_start = int((w - crop_size) / 2)

    patch_img = input_img[x_start:x_start+crop_size, y_start:y_start+crop_size]

    patch_img = cv2.resize(patch_img, (resize, resize))
    # cv2.imshow('patch', patch_img)
    # cv2.waitKey(0)
    if normalize:
        patch_img = patch_img.astype(np.float64)
        patch_img = (patch_img - np.min(patch_img)) / (np.max(patch_img) - np.mean(patch_img))

    return patch_img


def defineModel(model_type):
    pretrain_model_str = model_type

    if model_type == 'mynet3_50':
        model_ft = mynet.mynet3(layers=[3, 4, 6, 3])
    elif model_type == 'mynet3_101' or model_type == 'test':
        model_ft = mynet.mynet3(layers=[3, 4, 23, 3])
    elif model_type == 'mynet3_150' or 'mynet3_150_l1':
        model_ft = mynet.mynet3(layers=[3, 8, 36, 3])
    elif model_type == 'mynet4_150':
        model_ft = mynet.mynet4()
    else:
        print('Network type {} not supported!'.format(model_type))
        sys.exit()
    
    # model_path = path.join(model_folder, '3d_best_Generator_{}.pth'.format(pretrain_model_str))  # 10
    # model_ft.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    model_ft.cuda()
    model_ft.eval()
    model_ft = model_ft.to(device)

    return model_ft

# input an image array
# normalize values to 0-255
def array_normalize(input_array):
    max_value = np.max(input_array)
    min_value = np.min(input_array)
    # print('max {}, min {}'.format(max_value, min_value))
    k = 255 / (max_value - min_value)
    min_array = np.ones_like(input_array) * min_value
    normalized = k * (input_array - min_array)
    return normalized


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


def normalize_volume(input_volume):
    # print('input_volume shape {}'.format(input_volume.shape))
    mean = np.mean(input_volume)
    std = np.std(input_volume)

    normalized_volume = (input_volume - mean) / std
    # print('normalized shape {}'.format(normalized_volume.shape))
    # time.sleep(30)
    return normalized_volume


def scale_volume(input_volume, upper_bound=255, lower_bound=0):
    max_value = np.max(input_volume)
    min_value = np.min(input_volume)

    k = (upper_bound - lower_bound) / (max_value - min_value)
    scaled_volume = k * (input_volume - min_value) + lower_bound
    # print('min of scaled {}'.format(np.min(scaled_volume)))
    # print('max of scaled {}'.format(np.max(scaled_volume)))
    return scaled_volume

def chooseFrame(frame_num, mid_ratio=0.5):
    mid_ratio = 0.5
    frame_start = int(frame_num * (1 -  mid_ratio) // 2)
    frame_end = int(frame_num * (1 + mid_ratio) // 2)
    # print('start {}, end {}'.format(frame_start, frame_end))
    random_id = random.randint(frame_start, frame_end) - 1
    # print(random_id)

    return random_id



def get_dist_loss(labels, outputs, start_params, calib_mat):
    # print('labels shape {}'.format(labels.shape))
    # print('outputs shape {}'.format(outputs.shape))
    # print('start_params shape {}'.format(start_params.shape))
    # print('calib_mat shape {}'.format(calib_mat.shape))

    # print('labels_before\n{}'.format(labels.shape))
    labels = labels.data.cpu().numpy()
    outputs = outputs.data.cpu().numpy()
    if normalize_dof:
        labels = labels / dof_stats[:, 1] + dof_stats[:, 0]
        outputs = outputs / dof_stats[:, 1] + dof_stats[:, 0]


    start_params = start_params.data.cpu().numpy()
    calib_mat = calib_mat.data.cpu().numpy()

    if args.output_type == 'sum_dof':
        batch_errors = []
        for sample_id in range(labels.shape[0]):
            gen_param = tools.get_next_pos(trans_params1=start_params[sample_id, :],
                                           dof=outputs[sample_id, :],
                                           cam_cali_mat=calib_mat[sample_id, :, :])
            gt_param = tools.get_next_pos(trans_params1=start_params[sample_id, :],
                                          dof=labels[sample_id, :],
                                          cam_cali_mat=calib_mat[sample_id, :, :])
            gen_param = np.expand_dims(gen_param, axis=0)
            gt_param = np.expand_dims(gt_param, axis=0)

            result_pts = tools.params2corner_pts(params=gen_param, cam_cali_mat=calib_mat[sample_id, :, :])
            gt_pts = tools.params2corner_pts(params=gt_param, cam_cali_mat=calib_mat[sample_id, :, :])

            sample_error = tools.evaluate_dist(pts1=gt_pts, pts2=result_pts)
            batch_errors.append(sample_error)
        batch_errors = np.asarray(batch_errors)

        avg_batch_error = np.asarray(np.mean(batch_errors))
        error_tensor = torch.tensor(avg_batch_error, requires_grad=True)
        error_tensor = error_tensor.type(torch.FloatTensor)
        error_tensor = error_tensor.to(device)
        error_tensor = error_tensor * 0.99
        # print('disloss device {}'.format(device))
        # print(error_tensor)
        # time.sleep(30)
        return error_tensor




    if args.output_type == 'average_dof':
        labels = np.expand_dims(labels, axis=1)
        labels = np.repeat(labels, args.neighbour_slice - 1, axis=1)
        outputs = np.expand_dims(outputs, axis=1)
        outputs = np.repeat(outputs, args.neighbour_slice - 1, axis=1)
    else:
        labels = np.reshape(labels, (labels.shape[0], labels.shape[1] // 6, 6))
        outputs = np.reshape(outputs, (outputs.shape[0], outputs.shape[1] // 6, 6))
    # print('labels_after\n{}'.format(labels.shape))
    # print('outputs\n{}'.format(outputs.shape))
    # time.sleep(30)

    batch_errors = []
    final_drifts = []
    for sample_id in range(labels.shape[0]):
        gen_param_results = []
        gt_param_results = []
        for neighbour in range(labels.shape[1]):
            if neighbour == 0:
                base_param_gen = start_params[sample_id, :]
                base_param_gt = start_params[sample_id, :]
            else:
                base_param_gen = gen_param_results[neighbour - 1]
                base_param_gt = gt_param_results[neighbour - 1]
            gen_dof = outputs[sample_id, neighbour, :]
            gt_dof = labels[sample_id, neighbour, :]
            gen_param = tools.get_next_pos(trans_params1=base_param_gen, dof=gen_dof,
                                           cam_cali_mat=calib_mat[sample_id, :, :])
            gt_param = tools.get_next_pos(trans_params1=base_param_gt, dof=gt_dof,
                                          cam_cali_mat=calib_mat[sample_id, :, :])
            gen_param_results.append(gen_param)
            gt_param_results.append(gt_param)
        gen_param_results = np.asarray(gen_param_results)
        gt_param_results = np.asarray(gt_param_results)
        # print('gen_param_results shape {}'.format(gen_param_results.shape))

        result_pts = tools.params2corner_pts(params=gen_param_results, cam_cali_mat=calib_mat[sample_id, :, :])
        gt_pts = tools.params2corner_pts(params=gt_param_results, cam_cali_mat=calib_mat[sample_id, :, :])
        # print(result_pts.shape, gt_pts.shape)
        # time.sleep(30)

        results_final_vec = np.mean(result_pts[-1, :, :], axis=0)
        gt_final_vec = np.mean(gt_pts[-1, :, :], axis=0)
        final_drift = np.linalg.norm(results_final_vec - gt_final_vec) * 0.2
        final_drifts.append(final_drift)
        # print(results_final_vec, gt_final_vec)
        # print(final_drift)
        # time.sleep(30)

        sample_error = tools.evaluate_dist(pts1=gt_pts, pts2=result_pts)
        batch_errors.append(sample_error)

    batch_errors = np.asarray(batch_errors)
    avg_batch_error = np.asarray(np.mean(batch_errors))

    error_tensor = torch.tensor(avg_batch_error, requires_grad=True)
    error_tensor = error_tensor.type(torch.FloatTensor)
    error_tensor = error_tensor.to(device)
    error_tensor = error_tensor * 0.99
    # print('disloss device {}'.format(device))
    # print(error_tensor)
    # time.sleep(30)

    avg_final_drift = np.asarray(np.mean(np.asarray(final_drifts)))
    final_drift_tensor = torch.tensor(avg_final_drift, requires_grad=True)
    final_drift_tensor = final_drift_tensor.type(torch.FloatTensor)
    final_drift_tensor = final_drift_tensor.to(device)
    final_drift_tensor = final_drift_tensor * 0.99
    return error_tensor, final_drift_tensor


def get_correlation_loss(labels, outputs):
    # print('labels shape {}, outputs shape {}'.format(labels.shape, outputs.shape))
    x = outputs.flatten()
    y = labels.flatten()
    # print('x shape {}, y shape {}'.format(x.shape, y.shape))
    # print('x shape\n{}\ny shape\n{}'.format(x, y))
    xy = x * y
    mean_xy = torch.mean(xy)
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    cov_xy = mean_xy - mean_x * mean_y
    # print('xy shape {}'.format(xy.shape))
    # print('xy {}'.format(xy))
    # print('mean_xy {}'.format(mean_xy))
    # print('cov_xy {}'.format(cov_xy))

    var_x = torch.sum((x - mean_x) ** 2 / x.shape[0])
    var_y = torch.sum((y - mean_y) ** 2 / y.shape[0])
    # print('var_x {}'.format(var_x))

    corr_xy = cov_xy / (torch.sqrt(var_x * var_y))
    # print('correlation_xy {}'.format(corr_xy))

    loss = 1 - corr_xy
    # time.sleep(30)
    # x = output
    # y = target
    #
    # vx = x - torch.mean(x)
    # vy = y - torch.mean(y)
    #
    # loss = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    # print('correlation loss {}'.format(loss))
    # time.sleep(30)
    return loss



#

# ----- #
def _get_random_value(r, center, hasSign):
    randNumber = random.random() * r + center


    if hasSign:
        sign = random.random() > 0.5
        if sign == False:
            randNumber *= -1

    return randNumber


# ----- #
def get_array_from_itk_matrix(itk_mat):
    mat = np.reshape(np.asarray(itk_mat), (3, 3))
    return mat


# ----- #
def create_transform(aX, aY, aZ, tX, tY, tZ, mat_base=None):
    if mat_base is None:
        mat_base = np.identity(3)

    t_all = np.asarray((tX, tY, tZ))

    # Get the transform
    rotX = sitk.VersorTransform((1, 0, 0), aX / 180.0 * np.pi)
    matX = get_array_from_itk_matrix(rotX.GetMatrix())
    #
    rotY = sitk.VersorTransform((0, 1, 0), aY / 180.0 * np.pi)
    matY = get_array_from_itk_matrix(rotY.GetMatrix())
    #
    rotZ = sitk.VersorTransform((0, 0, 1), aZ / 180.0 * np.pi)
    matZ = get_array_from_itk_matrix(rotZ.GetMatrix())

    # Apply all the rotations
    mat_all = matX.dot(matY.dot(matZ.dot(mat_base[:3, :3])))

    return mat_all, t_all




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

def update_info(best_epoch, current_epoch, lowest_val_TRE):
    # readFile = open(os.path.join(output_dir, '{}.txt'.format(now_str)), "w")
    # lines = readFile.readlines()
    # readFile.close()

    file = open(os.path.join(output_dir, '{}.txt'.format(now_str)), 'a')
    file.write('Best_epoch: {}/{}\n'.format(best_epoch, current_epoch))
    file.write('Val_loss: {:.4f}\n'.format(lowest_val_TRE))
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
    if platform.system() == "Linux":
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
            tfm_RegS2V_gt_mat_tensor = torch.from_numpy(tfm_RegS2V_gt_pytorch) 


            """original transform_ITK (initial one)"""
            tfm_RegS2V_correction_pre = sitk.ReadTransform(data[key][3]) # transform (from parent) LPS
            tfm_RegS2V_correction_np_pre = ConvertITKTransform2NumpyArray(tfm_RegS2V_correction_pre) # from parent
            tfm_RegS2V_initial_np_from = tfm_3DUS_CT_np@tfm_RegS2V_initial_np@tfm_RegS2V_correction_np_pre # from parent
            tfm_RegS2V_initial_np_to = np.linalg.inv(tfm_RegS2V_initial_np_from) # (initial transform) to parent
            # print("initial transform:", tfm_RegS2V_initial_np_to)
            """transform_ITK to fake the pytorch transform (rescaled and recentered the orignal one)"""
            tfm_RegS2V_initial_np_to = T_scale@T_translate@tfm_RegS2V_initial_np_to@T_scale_inv  # this is the affine_transform_ITK
            """get transform_pytorch"""
            tfm_RegS2V_initial_pytorch = transform_conversion_ITK_to_pytorch(tfm_RegS2V_initial_np_to, self.volume_size) # this is the affine_transform_pytorch
            tfm_RegS2V_initial_mat_tensor = torch.from_numpy(tfm_RegS2V_initial_pytorch)
            """caculate the different between the initial one and ground truth, which is for training"""

            # # """this is based on the transform_ITK, we decided to use the transform_pytorch to train our model, see below"""
            # tfm_gt_diff_mat = tfm_RegS2V_gt_np_to@np.linalg.inv(tfm_RegS2V_initial_np_to)
            # tfm_gt_diff_mat_tensor = torch.from_numpy(tfm_gt_diff_mat)
            # tfm_gt_diff_dof = tools.mat2dof_np(input_mat=tfm_gt_diff_mat)
            # tfm_gt_diff_dof_tensor = torch.from_numpy(tfm_gt_diff_dof[:6])
            # # tfm_RegS2V_initial_mat_tensor = torch.from_numpy(tfm_RegS2V_initial_np_to)

            tfm_gt_diff_mat = tfm_RegS2V_gt_pytorch@np.linalg.inv(tfm_RegS2V_initial_pytorch)
            tfm_gt_diff_mat_tensor = torch.from_numpy(tfm_gt_diff_mat)
            
            tfm_gt_diff_dof = tools.mat2dof_np(input_mat=tfm_gt_diff_mat)
            tfm_gt_diff_dof_tensor = torch.from_numpy(tfm_gt_diff_dof[:6])
            

            data["tfm_gt_diff_mat"] = tfm_gt_diff_mat_tensor
            data["tfm_gt_diff_dof"] = tfm_gt_diff_dof_tensor
            data["tfm_RegS2V_initial_mat"] = tfm_RegS2V_initial_mat_tensor
            data["tfm_RegS2V_gt_mat"] = tfm_RegS2V_gt_mat_tensor


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
    loss_localNCC = LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size= 13, kernel_type="rectangular", reduction="mean")
    
    if args.training_mode == 'finetune':
        # overwrite the learning rate for finetune
        lr = 5e-6
        print('Learning rate is overwritten to be {}'.format(lr))
    else:
        lr = args.learning_rate
        print('Learning rate = {}'.format(lr))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    
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
                for i, batch in enumerate(training_dataloader_2DUS):
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

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        
                        optimizer.zero_grad()

                        vol_resampled, dof_estimated = model(vol=vol_tensor, frame=frame_tensor, initial_transform = batch['tfm_RegS2V_initial_mat'] ,device=device) # shape batch_size*6
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
                        

                        frame_tensor_4d = frame_tensor.squeeze(2)
                        # print("frame_tensor shape: ", frame_tensor.shape)
                        frame_tensor_gt = torch.zeros((frame_tensor_4d.shape))
                        for i, frame_flip_flag in enumerate(batch["frame_flip_flag"]):
                            if frame_flip_flag == "True":
                                frame_tensor_gt[i, :, :, :] = torch.flip(frame_tensor_4d[i, :, :, :], [2])
                        frame_tensor_gt = frame_tensor_gt.type(torch.FloatTensor).to(device)
                        
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
                        
                        image_localNCC_loss = loss_localNCC(frame_estimated, frame_tensor_gt)

                        # coefficients for loss functinos
                        alpha = 1.0
                        beta = 2.0
                        gamma = 5.0
                        loss_combined = alpha*rotation_loss + beta*translation_loss + gamma*image_localNCC_loss
                        
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

                    print('{}/{}: Train-BATCH: {:.4f}(loss_combined), {:.4f}(image_localNCC_loss), {:.4f}(loss_dof)'.format(nth_batch, math.ceil(num_cases[phase]/2), loss_combined, image_localNCC_loss, rotation_loss+translation_loss))
                
                scheduler.step()
                # sys.exit()
                epoch_loss = running_loss / num_cases[phase]
                epoch_running_localNCC = running_localNCC/num_cases[phase]
                epoch_running_dof = running_dof/num_cases[phase]
                tv_hist[phase].append([float(epoch_loss), float(epoch_running_localNCC), float(epoch_running_dof)])
                # print('tv_hist\n{}'.format(tv_hist))
                
            else:
                model.eval()

                for batch in validation_dataloader_2DUS:
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

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        
                        
                        vol_resampled, dof_estimated = model(vol=vol_tensor, frame=frame_tensor, initial_transform = batch['tfm_RegS2V_initial_mat'] ,device=device) # shape batch_size*6
                        
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
                       

                        frame_tensor_4d = frame_tensor.squeeze(2)
                        # print("frame_tensor shape: ", frame_tensor.shape)
                        frame_tensor_gt = torch.zeros((frame_tensor_4d.shape))
                        for i, frame_flip_flag in enumerate(batch["frame_flip_flag"]):
                            if frame_flip_flag == "True":
                                frame_tensor_gt[i, :, :, :] = torch.flip(frame_tensor_4d[i, :, :, :], [2])
                        frame_tensor_gt = frame_tensor_gt.type(torch.FloatTensor).to(device)
                 
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
                        
                        image_localNCC_loss = loss_localNCC(frame_estimated, frame_tensor_gt)

                        # coefficients for loss functinos
                        alpha = 1.0
                        beta = 2.0
                        gamma = 5.0
                        loss_combined = alpha*rotation_loss + beta*translation_loss + gamma*image_localNCC_loss
                        
                        # print("loss_combined is leaf_variable (guess False): ", loss_combined.is_leaf)
                        # print("loss_combined is required_grad (guess True): ", loss_combined.requires_grad)
                        # print("loss_combined device (guess cuda): ", loss_combined.device)
                        
                        # print("loss_combined: ", loss_combined)
                            
                        
                    # sys.exit()
                    running_loss += loss_combined * actual_batch_size
                    running_localNCC += image_localNCC_loss*actual_batch_size
                    running_dof += (rotation_loss + translation_loss)*actual_batch_size  
                
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
        update_info(best_epoch=best_ep+1, current_epoch=epoch+1, lowest_val_TRE=lowest_loss)
        print("=========================================================================")
        print('{}/{}: Train: {:.4f}(loss_combined), {:.4f}(loss_localNCC), {:.4f}(loss_dof), Validation: {:.4f}(loss_combined), {:.4f}(loss_localNCC), {:.4f}(loss_dof)'.format(
            epoch + 1, num_epochs,
            tv_hist['train'][-1][0],tv_hist['train'][-1][1], tv_hist['train'][-1][2],
            tv_hist['val'][-1][0], tv_hist['val'][-1][1], tv_hist['val'][-1][2]))
        print("=========================================================================")
        # sys.exit()
        
    time_elapsed = time.time() - since
    print('*' * 10 + 'Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return tv_hist


if __name__ == '__main__':
    
    """load dataset setting file and create the dictionary"""
    
    # project_dir = 'E:\PROGRAM\Project_PhD\Registration\DeepRegS2V'
    project_dir = os.getcwd() # Note: direct to the project folder
    data_tree_file = os.path.join(project_dir, "src/dataset_index_test.xml")
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
    resample_spacing = 0.5
    resize_scale = 1/resample_spacing
    volume_size = [400, 320, 240]

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
            LoadRegistrationTransformd(keys=["tfm_RegS2V"], scale=2, volume_size=volume_size),
            LoadImaged(keys=["frame_name", "frame_mask_name"], reader=ITKReader(reverse_indexing=False, affine_lps_to_ras=False), image_only=False),
            # MaskIntensityd(keys=["image","image_mask"], mask_data= mask_image_array),
            MaskIntensityd(keys=["frame_name"], mask_key= "frame_mask_name"),
            EnsureChannelFirstd(keys = ["frame_name"]),
            Spacingd(keys=["frame_name"], pixdim=(resample_spacing, resample_spacing, resample_spacing), mode=("bilinear")),
            SpatialPadd(keys=["frame_name"], spatial_size=[volume_size[0], volume_size[1], 1], method="symmetric", mode="constant"),
            CenterSpatialCropd(keys=["frame_name"], roi_size=[volume_size[0], volume_size[1], 1]), # when the spacing is 0.5*0.5*0.5
            ScaleIntensityd(keys=["frame_name"], minv= 0.0, maxv = 1.0, dtype= np.float32)
        ]
    )
    

    training_dataset_3DUS = CacheDataset(data=training_volume_dict, transform=transform_3DUS)
    training_dataset_2DUS = CacheDataset(data=training_dataset_dict, transform=transform_2DUS)
    training_dataloader_2DUS = DataLoader(dataset=training_dataset_2DUS, batch_size=2, shuffle=False)

    validation_dataset_3DUS = CacheDataset(data=validation_volume_dict, transform=transform_3DUS)
    validation_dataset_2DUS = CacheDataset(data=validation_dataset_dict, transform=transform_2DUS)
    validation_dataloader_2DUS = DataLoader(dataset=validation_dataset_2DUS, batch_size=2, shuffle=False)

    # print("number of cases: ", len(training_dataset_2DUS))
    # sys.exit()
    num_cases = {'train': len(training_dataset_2DUS), 'val': len(validation_dataset_2DUS)}
    # Define the training model architecture

    model = RegS2Vnet.mynet3(layers=[3, 8, 36, 3]).to(device=device)
    train_model(model=model, training_dataset_frame=training_dataloader_2DUS, training_dataset_volume = training_dataset_3DUS, validation_dataset_frame = validation_dataloader_2DUS, validation_dateset_volume = validation_dataset_3DUS, num_cases= num_cases)
