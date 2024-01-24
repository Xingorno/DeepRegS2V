# Demo image registration using SimpleITK

from matplotlib import pyplot as plt
import numpy as np
import SimpleITK as sitk
import time
import pandas as pd
from os import path
import os
import sys
import cv2
import imageio
import torch
import torchgeometry as tgm
import math
from utils import transformations as tfms
import random

def normalize_vector(v):
    """v: batch*n
    output: batch*n"""
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v

def cross_product(u, v):
    """u, v: batch*3
    output: batch*3"""
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
    return out

def compute_rotation_matrix_from_ortho6d(poses):
    """poses: batch*6
    output: batch*3*3 """
    x_raw = poses[:,0:3]#batch*3
    y_raw = poses[:,3:6]#batch*3
        
    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z)#batch*3
    y = cross_product(z,x)#batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix

def compute_rotation_matrix_from_matrix(matrices):
    """matrices: batch*3*3
    output: batch*3*3"""
    b = matrices.shape[0]
    a1 = matrices[:,:,0]#batch*3
    a2 = matrices[:,:,1]
    a3 = matrices[:,:,2]
    
    u1 = a1
    u2 = a2 - proj_u_a(u1,a2)
    u3 = a3 - proj_u_a(u1,a3) - proj_u_a(u2,a3)
    
    e1 = normalize_vector(u1)
    e2 = normalize_vector(u2)
    e3 = normalize_vector(u3)
    
    rmat = torch.cat((e1.view(b, 3,1), e2.view(b,3,1),e3.view(b,3,1)), 2)
    
    return rmat


def proj_u_a(u,a):
    """u, a: batch*3
    output: batch*3"""
    batch=u.shape[0]
    top = u[:,0]*a[:,0] + u[:,1]*a[:,1]+u[:,2]*a[:,2]
    bottom = u[:,0]*u[:,0] + u[:,1]*u[:,1]+u[:,2]*u[:,2]
    bottom = torch.max(torch.autograd.Variable(torch.zeros(batch).cuda())+1e-8, bottom)
    factor = (top/bottom).view(batch,1).expand(batch,3)
    out = factor* u
    return out

def compute_rotation_matrix_from_quaternion(quaternion):
    """quaternion: batch*4;
    output: batch*3*3"""
    batch=quaternion.shape[0]
    
    quat = normalize_vector(quaternion)
    
    qw = quat[...,0].view(batch, 1)
    qx = quat[...,1].view(batch, 1)
    qy = quat[...,2].view(batch, 1)
    qz = quat[...,3].view(batch, 1)

    # Unit quaternion rotation matrices computatation  
    xx = qx*qx
    yy = qy*qy
    zz = qz*qz
    xy = qx*qy
    xz = qx*qz
    yz = qy*qz
    xw = qx*qw
    yw = qy*qw
    zw = qz*qw
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    
    return matrix

def compute_rotation_matrix_from_axisAngle(axisAngle):
    """axisAngle: batch*4 (angle, x, y, z);
    output: batch*3*3"""
    batch = axisAngle.shape[0]
    
    theta = torch.tanh(axisAngle[:,0])*np.pi #[-180, 180]
    sin = torch.sin(theta)
    axis = normalize_vector(axisAngle[:,1:4]) #batch*3
    qw = torch.cos(theta)
    qx = axis[:,0]*sin
    qy = axis[:,1]*sin
    qz = axis[:,2]*sin
    
    # Unit quaternion rotation matrices computatation  
    xx = (qx*qx).view(batch,1)
    yy = (qy*qy).view(batch,1)
    zz = (qz*qz).view(batch,1)
    xy = (qx*qy).view(batch,1)
    xz = (qx*qz).view(batch,1)
    yz = (qy*qz).view(batch,1)
    xw = (qx*qw).view(batch,1)
    yw = (qy*qw).view(batch,1)
    zw = (qz*qw).view(batch,1)
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    
    return matrix

def compute_rotation_matrix_from_euler(euler):
    """ euler: batch*3
    rotation order: matrix = rotx(ai) * roty(aj) * rotz(ak)"""
    """euler unit: rad """
    # print("input_dof is leaf_variable (guess false): ", input_dof.is_leaf)
    # print("input_dof is required_grad (guess True): ", input_dof.requires_grad)

    
    ai = euler[:, 0]
    aj = euler[:, 1]
    ak = euler[:, 2]

    si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
    ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)

    M = torch.zeros((euler.shape[0], 3, 3))
    
    M[:, 0, 0] = cj*ck
    M[:, 0, 1] = -cj*sk
    M[:, 0, 2] = sj
    M[:, 1, 0] = si*sj*ck + ci*sk
    M[:, 1, 1] = -si*sj*sk + ci*ck
    M[:, 1, 2] = -si*cj
    M[:, 2, 0] = -ci*sj*ck + si*sk
    M[:, 2, 1] = ci*sj*sk + si*ck
    M[:, 2, 2] = ci*cj
    
    return M

def compute_geodesic_distance_from_two_matrices(m1, m2):
    """m1, m2: batch*3*3
    output: between 0 to pi/2 radian batch"""
    batch=m1.shape[0]
    # print("m1 device {}".format(m1.device))
    # print("m2 transpose {}".format(m2.transpose(1,2).device))
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
    
    cos = (m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda())*-1 )
    
    
    theta = torch.acos(cos)
    
    #theta = torch.min(theta, 2*np.pi - theta)
    
    
    return theta


def networkOutput2AffineTransform_pytorch(rotation_para, translation_para,mode= "ortho6d", vol_size = [200, 160, 120], device = None):
    """rotation_para: batch*3, batch*4, batch*6, batch*9
    translation_para: batch*3 (already normalized)
    mode: "ortho6d", "quaternion", "rmat", "euler", or "axisAngle"
    vol_size: [3'] the volume size
    output: normalized transformation batch*4*4 for pytorch affine_grid
    unnormalized transformation batch*3*3"""
    if mode == "ortho6d":
        rotation_para_n6 = rotation_para
        rot_m = compute_rotation_matrix_from_ortho6d(rotation_para_n6) # batch*3*3
        # print(mode)
    elif mode == "quaternion":
        rotation_para_n4 = rotation_para
        rot_m = compute_rotation_matrix_from_quaternion(rotation_para_n4)
        # print(mode)
    elif mode == "rmat":
        rotation_para_n33 = rotation_para.view(-1, 3, 3) # batch*9 => batch*3*3
        rot_m = compute_rotation_matrix_from_matrix(rotation_para_n33)
        # print(mode)
    elif mode == "euler":
        rotation_para_n3 = rotation_para
        rot_m = compute_rotation_matrix_from_euler(rotation_para_n3)
        # print(mode)
    elif mode == "axisAngle":
        rotation_para_n4 = rotation_para
        rot_m = compute_rotation_matrix_from_axisAngle(rotation_para_n4)
        # print(mode)
    rot_m = rot_m.to(device=device)
    # normalize rotation matrix
    T_normalized = torch.tensor([[2/vol_size[0], 0.0, 0.0], [0.0, 2/vol_size[1], 0.0], [0.0, 0.0, 2/vol_size[2]]]).type(torch.FloatTensor)
    T_normalized = T_normalized.to(device)
    T_normalized_inv = torch.linalg.inv(T_normalized).type(torch.FloatTensor)
    T_normalized_inv = T_normalized_inv.to(device)
    rot_m_normalized = T_normalized @ rot_m @ T_normalized_inv

    # combine the rotation matrix with the transaltion matrix
    rt_mat = torch.zeros((rotation_para.shape[0], 4, 4))
    rt_mat = rt_mat.to(device=device)
    rt_mat[:,0:3, 0:3] = rot_m_normalized
    rt_mat[:, 3, 3] = 1
    rt_mat[:, :3, 3] = translation_para

    return rt_mat


def networkOutput2AffineTransform_ITK(rotation_para, translation_para, mode= "ortho6d", vol_size = [200, 160, 120], device = None):
    """rotation_para: batch*3, batch*4, batch*6, batch*9
    translation_para: batch*3 (already normalized)
    mode: "ortho6d", "quaternion", "rmat", "euler", or "axisAngle"
    vol_size: [3'] the volume size
    output: normalized transformation batch*4*4 for ITK"""
    if mode == "ortho6d":
        rotation_para_n6 = rotation_para
        rot_m = compute_rotation_matrix_from_ortho6d(rotation_para_n6) # batch*3*3
        # print(mode)
    elif mode == "quaternion":
        rotation_para_n4 = rotation_para
        rot_m = compute_rotation_matrix_from_quaternion(rotation_para_n4)
        # print(mode)
    elif mode == "rmat":
        rotation_para_n33 = rotation_para.view(-1, 3, 3) # batch*9 => batch*3*3
        rot_m = compute_rotation_matrix_from_matrix(rotation_para_n33)
        # print(mode)
    elif mode == "euler":
        rotation_para_n3 = rotation_para
        rot_m = compute_rotation_matrix_from_euler(rotation_para_n3)
        # print(mode)
    elif mode == "axisAngle":
        rotation_para_n4 = rotation_para
        rot_m = compute_rotation_matrix_from_axisAngle(rotation_para_n4)
        # print(mode)
    
    # denormalize the translation matrix
    T_denormalize = torch.tensor([[vol_size[0]*0.5, vol_size[1]*0.5, vol_size[2]*0.5]])
    T_denormalize = T_denormalize.to(device=device)
    translation_denormalized = translation_para*T_denormalize
    # print("translation unnormalized: {}".format(translation_denormalized))
    # combine the rotation matrix with the transaltion matrix
    rt_mat_denormalized_inv = torch.zeros((rotation_para.shape[0], 4, 4))
    rt_mat_denormalized_inv = rt_mat_denormalized_inv.to(device=device)
    rt_mat_denormalized_inv[:,0:3, 0:3] = rot_m
    rt_mat_denormalized_inv[:, 3, 3] = 1
    rt_mat_denormalized_inv[:, :3, 3] = translation_denormalized
    rt_mat_denormalized = torch.linalg.inv(rt_mat_denormalized_inv) # important
    return rt_mat_denormalized, rt_mat_denormalized_inv
    

def mat2tfm(input_mat):
    tfm = sitk.AffineTransform(3)
    tfm.SetMatrix(np.reshape(input_mat[:3, :3], (9,)))
    translation = input_mat[:3,3]
    tfm.SetTranslation(translation)
    # tfm.SetCenter([0, 0, 0])
    return tfm


def computeScale(input_mat):
    scale1 = np.linalg.norm(input_mat[:3, 0])
    scale2 = np.linalg.norm(input_mat[:3, 1])
    scale3 = np.linalg.norm(input_mat[:3, 2])
    # print('scale1 {}'.format(scale1))
    # print('scale2 {}'.format(scale2))
    # print('scale3 {}'.format(scale3))
    # print(0.478425 * 0.35)
    # sys.exit()
    return np.asarray([scale1, scale2, scale3])



def mat2dof_np(input_mat):
    # print('input_mat\n{}'.format(input_mat))
    translations = input_mat[:3, 3]
    rotations_eulers = np.asarray(tfms.euler_from_matrix(input_mat, 'rxyz'))
    rotations_degrees = (rotations_eulers / (2 * math.pi)) * 360
    scales = computeScale(input_mat=input_mat)

    dof = np.concatenate((translations, rotations_degrees, scales), axis=0)

    # print('dof\n{}\n'.format(dof))
    # sys.exit()
    return dof

def dof2mat_np(input_dof, scale=False):
    """ Transfer degrees to euler """
    dof = input_dof
    
    dof[3:6] = dof[3:6] *  math.pi / 180.0
    # print('rad {}'.format(dof[3:6]))


    rot_mat = tfms.euler_matrix(dof[3], dof[4], dof[5], 'rxyz')[:3, :3]

    mat44 = np.identity(4)
    mat44[:3, :3] = rot_mat
    mat44[:3, 3] = dof[:3]

    if scale:
        scales = dof[6:]
        mat_scale = np.diag([scales[1], scales[0], scales[2], 1])
        mat44 = np.dot(mat44, np.linalg.inv(mat_scale))
    # print('mat_scale\n{}'.format(mat_scale))
    # print('recon mat\n{}'.format(mat44))
    # sys.exit()
    return mat44


def dof2mat_tensor(input_dof):
    """rotation order: matrix = rotx(ai) * roty(aj) * rotz(ak)"""

    # print("input_dof is leaf_variable (guess false): ", input_dof.is_leaf)
    # print("input_dof is required_grad (guess True): ", input_dof.requires_grad)

    rad = tgm.deg2rad(input_dof[:, 3:])

    ai = rad[:, 0]
    aj = rad[:, 1]
    ak = rad[:, 2]

    si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
    ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)

    M = torch.zeros((input_dof.shape[0], 4, 4))
    
    M[:, 0, 0] = cj*ck
    M[:, 0, 1] = -cj*sk
    M[:, 0, 2] = sj
    M[:, 1, 0] = si*sj*ck + ci*sk
    M[:, 1, 1] = -si*sj*sk + ci*ck
    M[:, 1, 2] = -si*cj
    M[:, 2, 0] = -ci*sj*ck + si*sk
    M[:, 2, 1] = ci*sj*sk + si*ck
    M[:, 2, 2] = ci*cj
    M[:, 3, 3] = 1
    M[:, :3, 3] = input_dof[:, :3]

    
    
    # print("M is leaf_variable (guess false): ", M.is_leaf)
    # print("M is required_grad (guess True): ", M.requires_grad)

    # print('out_mat {}\n{}'.format(M.shape, M))
    # sys.exit()
    return M

def dof2mat_tensor_normalized(input_dof):
    """rotation order: matrix = rotx(ai) * roty(aj) * rotz(ak)"""
    """translation: normalized by volume_size"""
    """rotation dof: rad """
    # print("input_dof is leaf_variable (guess false): ", input_dof.is_leaf)
    # print("input_dof is required_grad (guess True): ", input_dof.requires_grad)

    # rad = tgm.deg2rad(input_dof[:, 3:])
    rad = input_dof[:, 3:]
    ai = rad[:, 0]
    aj = rad[:, 1]
    ak = rad[:, 2]

    si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
    ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)

    M = torch.zeros((input_dof.shape[0], 4, 4))
    
    M[:, 0, 0] = cj*ck
    M[:, 0, 1] = -cj*sk
    M[:, 0, 2] = sj
    M[:, 1, 0] = si*sj*ck + ci*sk
    M[:, 1, 1] = -si*sj*sk + ci*ck
    M[:, 1, 2] = -si*cj
    M[:, 2, 0] = -ci*sj*ck + si*sk
    M[:, 2, 1] = ci*sj*sk + si*ck
    M[:, 2, 2] = ci*cj
    M[:, 3, 3] = 1
    M[:, :3, 3] = input_dof[:, :3]

    
    
    # print("M is leaf_variable (guess false): ", M.is_leaf)
    # print("M is required_grad (guess True): ", M.requires_grad)

    # print('out_mat {}\n{}'.format(M.shape, M))
    # sys.exit()
    return M



def mat2dof_tensor(input_mat, degree = 'deg'):
    transform_input_np = torch.Tensor.numpy(input_mat.detach().cpu())
    transform_shape = transform_input_np.shape
    # print("transform_shape: ", transform_shape)
    
    if len(transform_shape) == 3:
        input_dof_np = np.zeros((transform_shape[0], 6))
        for i in range(0, transform_shape[0]):
            input_dof_np[i,:] = mat2dof_np(transform_input_np[i])[0:6]
        input_dof_tensor = torch.from_numpy(input_dof_np)
        if degree == 'deg':
            input_dof_tensor
        if degree == 'rad':
            input_dof_tensor[:, 3:] = input_dof_tensor[:, 3:] * (math.pi) / 180.0
        return input_dof_tensor
    
    elif len(transform_shape) == 2:
        input_dof_np = np.zeros((1, 6))
        input_dof_np[0,:] = mat2dof_np(transform_input_np)[0:6]
        input_dof_tensor = torch.from_numpy(input_dof_np)
        if degree == 'deg':
            input_dof_tensor
        if degree == 'rad':
            input_dof_tensor[:, 3:] = input_dof_tensor[:, 3:] * (math.pi) / 180.0
        return input_dof_tensor
    else:
         print("input error about wrong transformation dimensionality")


def dof2mat_tensor_backup(input_dof, device):
    """Note: the order of rigid transformation (first: translation, secondly: rotation)"""
    input_dof_clone = input_dof.clone()
    
    rad = tgm.deg2rad(input_dof_clone[:, 3:])
    print('input_dof shape {}'.format(input_dof_clone.shape))
    ai = rad[:, 0]
    aj = rad[:, 1]
    ak = rad[:, 2]

    si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
    ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = torch.zeros((input_dof_clone.shape[0], 4, 4))
    
    print('M shape {}'.format(M.shape))
    print('M:{}'.format(M))
    if device:
        M = M.to(device)
        M.requires_grad = False # TODO: dont know why this needs to be differentiable

    M[:, 0, 0] = cj*ck
    M[:, 0, 1] = sj*sc-cs
    M[:, 0, 2] = sj*cc+ss
    M[:, 1, 0] = cj*sk
    M[:, 1, 1] = sj*ss+cc
    M[:, 1, 2] = sj*cs-sc
    M[:, 2, 0] = -sj
    M[:, 2, 1] = cj*si
    M[:, 2, 2] = cj*ci
    M[:, :3, 3] = input_dof_clone[:, :3]

    # print('out_mat {}\n{}'.format(M.shape, M))
    # sys.exit()
    return M



def correlation_coefficient(T1, T2):
    numerator = np.mean((T1 - T1.mean()) * (T2 - T2.mean()))
    denominator = T1.std() * T2.std()
    if denominator == 0:
        return 0
    else:
        result = numerator / denominator
        return result




