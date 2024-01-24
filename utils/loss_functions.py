import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import tools

def loss_geodesic(gt_rmat, predict_rmat):
    theta = tools.compute_geodesic_distance_from_two_matrices(gt_rmat, predict_rmat)
    # print("geodesic distance (radian): {}".format(theta))
    error = theta.mean()
    return error


class LocalNCC_new:
    def __init__(self, device = "cuda:0", kernel_size =(51, 51), stride = (1, 1), sampling = 'regular', padding = 'valid', reduction = 'mean', cropping = True, eps = 1e-5, win_eps = 0.98):
        
        self.kernel_size = kernel_size
        # if self.kernel_size == 'none':
        #     self.kernel_size = (21, 21)
        self.sampling = sampling
        self.reduction = reduction
        self.stride = stride
        self.padding = padding
        self.eps = eps
        self.cropping = cropping
        self.win_eps = win_eps
        self.device = device

    def __call__(self, image_predicted, image_target):
        """Preprocess the images, to avoid the sub-optimal problem"""
        """image_predicted: tensor, N*1*H*W; image_target: tensor, N*1*H*W"""
        # preprocessing the image (define the ROI using the intersection of image_predicted and image_target)
        loss_localNCC_total = 0
        
        ROI = []
        if self.cropping:

            image_predicted_mask = torch.zeros(image_predicted.shape)
            image_predicted_mask[image_predicted > 0.0] = 1
            image_predicted_mask = image_predicted_mask.to(self.device)    
            image_target_mask = torch.zeros(image_target.shape)
            image_target_mask[image_target > 0.0] = 1
            image_target_mask = image_target_mask.to(self.device)
            image_ROI = image_predicted_mask * image_target_mask # should be leaf = false, requires_grad = true
            
            # valid cropping
            sum_filter = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
            sum_filter = sum_filter.to(device=self.device)
            image_ROI_sum = F.conv2d(image_ROI, sum_filter, stride= 1, padding = "same")
            # print("image_ROI_sum shape: {}".format(image_ROI_sum.shape))
            image_ROI_valid_mask = torch.ones(image_ROI_sum.shape)
            image_ROI_valid_mask = image_ROI_valid_mask.to(device = self.device)
            threshold = int(self.kernel_size[0]*self.kernel_size[1] * self.win_eps)
            image_ROI_valid_mask = image_ROI_valid_mask * (image_ROI_sum > threshold)


            for i in range(image_predicted.shape[0]):
                indices = torch.nonzero(image_ROI[i, 0, :, :], as_tuple = True)
                # print("indices: ", indices)
                if indices[0].numel() == 0:
                    H_min, H_max = int(image_predicted.shape[2]*0.5 - 0.6*self.kernel_size[0]), int(image_predicted.shape[2]*0.5 + 0.6*self.kernel_size[0])
                    W_min, W_max = int(image_predicted.shape[3]*0.5 - 0.6*self.kernel_size[1]), int(image_predicted.shape[3]*0.5 + 0.6*self.kernel_size[1])
                else:
                    H_min, H_max = torch.min(indices[0]), torch.max(indices[0])
                    W_min, W_max = torch.min(indices[1]), torch.max(indices[1])

                    
                    if (H_max-H_min + 1) < self.kernel_size[0]:
                        H_min_ = torch.min(H_min, torch.floor(((H_max+H_min)*0.5 - self.kernel_size[0]*0.5 -1)).type(torch.int16))
                        H_max_ = torch.max(H_max, torch.ceil(((H_max+H_min)*0.5 + self.kernel_size[0]*0.5)).type(torch.int16))

                        if H_min_ < 1:
                            H_max = H_max_ - H_min_ +1
                            H_min = 1
                        elif H_max_ > image_predicted.shape[2]-1:
                            H_min = H_min_ - (H_max_ - image_predicted.shape[2]).type(torch.int16)
                            H_max = image_predicted.shape[2]-1
                        else:
                            H_max = H_max_
                            H_min = H_min_

                    if (W_max-W_min + 1) < self.kernel_size[1]:
                        W_min_ = torch.min(W_min, torch.floor(((W_max+W_min)*0.5- self.kernel_size[1]*0.5 -1)).type(torch.int16))
                        W_max_ = torch.max(W_max, torch.ceil(((W_max+W_min)*0.5+ self.kernel_size[1]*0.5)).type(torch.int16))
                        if W_min_ < 1:
                            W_max = W_max_ -W_min_ + 1
                            W_min = 1
                        elif W_max_ > image_predicted.shape[3]-1:
                            W_min = W_min_ - (W_max_ - image_predicted.shape[3]).type(torch.int16)
                            W_max = image_predicted.shape[3]-1
                        else:
                            W_min = W_min_
                            W_max = W_max_

                ROI_case =  {"H_min": H_min, "H_max": H_max, "W_min": W_min, "W_max": W_max}

                
                image_pred = image_predicted[i, :, H_min:H_max+ 1, W_min:W_max+1]
                image_tar = image_target[i, :, H_min:H_max+ 1, W_min:W_max+1]
                image_ROI_valid_mask_ = image_ROI_valid_mask[i, :, H_min:H_max+ 1, W_min:W_max+1].unsqueeze(0)

                loss_localNCC = self.run(image_pred, image_tar, image_ROI_valid_mask_)
                
                # print("loss_localNCC (device): ", loss_localNCC.device)
                loss_localNCC_total +=loss_localNCC
                ROI.append(ROI_case)

        else:
            for i in range(image_predicted.shape[0]):

                indices = torch.nonzero(image_ROI_valid_mask[i, 0, :, :], as_tuple = True)
                # print("indices: ", indices)
                if indices[0].numel() == 0:
                    H_min, H_max = image_predicted.shape[2]*0.5 - self.kernel_size, image_predicted.shape[2]*0.5 + self.kernel_size
                    W_min, W_max = image_predicted.shape[3]*0.5 - self.kernel_size, image_predicted.shape[3]*0.5 + self.kernel_size
                else:
                    H_min, H_max = torch.min(indices[0]), torch.max(indices[0])
                    W_min, W_max = torch.min(indices[1]), torch.max(indices[1])
                    if (H_max-H_min) < self.kernel_size[0]:
                        H_min = torch.min(H_min, (H_max+H_min)*0.5- self.kernel_size[0]*0.5)
                        H_max = torch.max(H_max, (H_max+H_min)*0.5+ self.kernel_size[0]*0.5)
                        if H_min < 0:
                            H_max = H_max - H_min
                            H_min = 0
                        if H_max > image_predicted.shape[2]-1:
                            H_min = H_min - (H_max - image_predicted.shape[2]+1)
                            H_max = image_predicted.shape[2]-1

                    if (W_max-W_min) < self.kernel_size[1]:
                        W_min = torch.min(W_min, (W_max+W_min)*0.5- self.kernel_size[1]*0.5)
                        W_max = torch.max(W_max, (W_max+W_min)*0.5+ self.kernel_size[1]*0.5)
                        if W_min < 0:
                            W_max = W_max -W_min
                            W_min = 0
                        if W_max > image_predicted.shape[3]-1:
                            W_min = W_min - (W_max - image_predicted.shape[3]+1)
                            W_max = image_predicted.shape[3]-1

                ROI_case =  {"H_min": H_min, "H_max": H_max, "W_min": W_min, "W_max": W_max}
                
                ROI.append(ROI_case)

        
        if self.reduction == 'mean':
            loss_localNCC_total = loss_localNCC_total/image_predicted.shape[0]

            loss_localNCC_total = loss_localNCC_total.type(torch.FloatTensor)
            loss_localNCC_total = loss_localNCC_total.to(self.device)

            
        if self.reduction == 'sum':
            # TODO: need to implement this
            loss_localNCC_total = loss_localNCC_total.type(torch.FloatTensor)
            loss_localNCC_total = loss_localNCC_total.to(self.device)
        
        return -loss_localNCC_total, ROI

    def run(self, image_pred, image_tar, image_ROI_valid_mask):
        """Calculate the local normalized cross correlation (default: dim =2)
        image_combined_pred_target: [B, C, H, W] """
        """
        Local squared zero-normalized cross-correlation.
        Denote y_true as t and y_pred as p. Consider a window having n elements.
        Each position in the window corresponds a weight w_i for i=1:n.
        Define the discrete expectation in the window E[t] as
            E[t] = sum_i(w_i * t_i) / sum_i(w_i)

        Similarly, the discrete variance in the window V[t] is
            V[t] = E[t**2] - E[t] ** 2

        The local squared zero-normalized cross-correlation is therefore
            E[ (t-E[t]) * (p-E[p]) ] ** 2 / V[t] / V[p]

        where the expectation in numerator is
            E[ (t-E[t]) * (p-E[p]) ] = E[t * p] - E[t] * E[p]

        Different kernel corresponds to different weights.
        Reference:
            - Zero-normalized cross-correlation (ZNCC):
                https://en.wikipedia.org/wiki/Cross-correlation
            - Code: https://github.com/voxelmorph/voxelmorph/blob/legacy/src/losses.py
        """
        
        pred = image_pred.unsqueeze(0)
        target = image_tar.unsqueeze(0)
        pred2 = pred*pred 
        target2 = target*target
        pred_target = pred*target

        # compute the sum filter
        sum_filter = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
        sum_filter = sum_filter.to(self.device)
        one_filter = torch.zeros(1, 1, self.kernel_size[0], self.kernel_size[1])
        one_filter[:, :, int((self.kernel_size[0]-1)*0.5), int((self.kernel_size[1]-1)*0.5)] = 1
        one_filter.type(torch.FloatTensor)
        one_filter = one_filter.to(self.device)

        pred_sum = F.conv2d(pred, sum_filter, stride= self.stride, padding = self.padding)
        target_sum = F.conv2d(target, sum_filter, stride = self.stride, padding = self.padding)
        pred_target_sum = F.conv2d(pred_target, sum_filter, stride = self.stride, padding = self.padding)


        pred2_sum = F.conv2d(pred2, sum_filter,stride = self.stride, padding = self.padding)
        target2_sum = F.conv2d(target2, sum_filter,stride = self.stride, padding = self.padding)

        image_ROI_valid_mask_filtered = F.conv2d(image_ROI_valid_mask, one_filter, stride=self.stride, padding=self.padding)

        # average over kernel
        # print("image_ROI_valid_mask_filtered requires_grad: {}".format(image_ROI_valid_mask_filtered.requires_grad))
        win_size = self.kernel_size[0] * self.kernel_size[1]
        u_pred = pred_sum/win_size
        u_target = target_sum/win_size

        cross = pred_target_sum - u_pred*target_sum
        target_var = torch.max(target2_sum - u_target * target_sum, torch.as_tensor(self.eps, dtype=target2_sum.dtype, device=target2_sum.device))
        pred_var = torch.max(pred2_sum - u_pred * pred_sum, torch.as_tensor(self.eps, dtype=pred2_sum.dtype, device=pred2_sum.device))
        
        # loss_localNCC = cross*cross / (target_var*pred_var)
        loss_localNCC = cross / (torch.sqrt(target_var*pred_var))

        # print("loss_localNCC dtype: {}".format(loss_localNCC.dtype))
        # # filter out the blank areas
        image_ROI_valid_mask_flatten = torch.flatten(image_ROI_valid_mask_filtered).to(device=loss_localNCC.device)
        nonzero_indices = torch.nonzero(image_ROI_valid_mask_flatten, as_tuple = False)

        loss_localNCC_flatten = torch.flatten(loss_localNCC)

        if nonzero_indices.numel() == 0:
            nonzero_indices = torch.Tensor([int(image_ROI_valid_mask_flatten.shape[0]*0.5)]).type(torch.int)

            loss_localNCC_mean = torch.mean(loss_localNCC_flatten[nonzero_indices]) / (int(image_ROI_valid_mask_flatten.shape[0]))
        else:

            loss_localNCC_mean = torch.mean(loss_localNCC_flatten[nonzero_indices])

        return loss_localNCC_mean

class LocalNCC:
    def __init__(self, device = "cpu", kernel_size =(51, 51), stride = (1, 1), sampling = 'regular', padding = 'valid', reduction = 'mean', cropping = True, eps = 1e-3, win_eps = 0.1):
        
        self.kernel_size = kernel_size
        # if self.kernel_size == 'none':
        #     self.kernel_size = (21, 21)
        self.sampling = sampling
        self.reduction = reduction
        self.stride = stride
        self.padding = padding
        self.eps = eps
        self.cropping = cropping
        self.win_eps = win_eps
        self.device = device

    def __call__(self, image_predicted, image_target):
        """Preprocess the images, to avoid the sub-optimal problem"""
        """image_predicted: tensor, N*1*H*W; image_target: tensor, N*1*H*W"""
        # preprocessing the image (define the ROI using the intersection of image_predicted and image_target)
        loss_localNCC_total = 0
        
        ROI = []
        if self.cropping:

            image_predicted_mask = torch.zeros(image_predicted.shape)
            image_predicted_mask[image_predicted > 0.0] = 1
            image_predicted_mask = image_predicted_mask.to(self.device)    
            image_target_mask = torch.zeros(image_target.shape)
            image_target_mask[image_target > 0.0] = 1
            image_target_mask = image_target_mask.to(self.device)
            image_ROI = image_predicted_mask * image_target_mask # should be leaf = false, requires_grad = true
            # print("image_ROI shape: ", image_ROI.shape)
            
            for i in range(image_predicted.shape[0]):
                indices = torch.nonzero(image_ROI[i, 0, :, :], as_tuple = True)
                # print("indices: ", indices)
                if indices[0].numel() == 0:
                    H_min, H_max = 0, image_predicted.shape[2]-1
                    W_min, W_max = 0, image_predicted.shape[3]-1
                else:
                    H_min, H_max = torch.min(indices[0]), torch.max(indices[0])
                    W_min, W_max = torch.min(indices[1]), torch.max(indices[1])
                    if (H_max-H_min) < self.kernel_size[0] or (W_max-W_min) < self.kernel_size[1]:
                        H_min, H_max = 0, image_predicted.shape[2]-1
                        W_min, W_max = 0, image_predicted.shape[3]-1 
                ROI_case =  {"H_min": H_min, "H_max": H_max, "W_min": W_min, "W_max": W_max}
                
                # ROIed
                image_combined_pred_target = torch.zeros(2, 1, H_max-H_min+1, W_max-W_min+1)
                
                # print(image_combined_pred_target.shape)
                # print("H_min: {}, H_max: {}, W_min: {}, W_max: {}".format(H_min, H_max, W_min, W_max))
                image_combined_pred_target[0,:,:,:] = image_predicted[i, :, H_min:H_max+ 1, W_min:W_max+1]
                image_combined_pred_target[1,:,:,:] = image_target[i, :, H_min:H_max+ 1, W_min:W_max+1]
                image_combined_pred_target = image_combined_pred_target.type(torch.FloatTensor).to(self.device)
                # print("image_combined_pred_target (require_grad): {}".format(image_combined_pred_target.requires_grad))
                loss_localNCC = self.run(image_combined_pred_target)
                # print("loss_localNCC (device): ", loss_localNCC.device)
                loss_localNCC_total +=loss_localNCC
                ROI.append(ROI_case)

        else:
            for i in range(image_predicted.shape[0]):

                image_combined_pred_target = torch.zeros(2, 1, image_predicted.shape[2], image_predicted.shape[3])
                H_min, H_max = 0, image_predicted.shape[2]-1
                W_min, W_max = 0, image_predicted.shape[3]-1
                ROI_case =  {"H_min": H_min, "H_max": H_max, "W_min": W_min, "W_max": W_max}
                
                # print(image_combined_pred_target.shape)
                # print("H_min: {}, H_max: {}, W_min: {}, W_max: {}".format(H_min, H_max, W_min, W_max))
                image_combined_pred_target[0,:,:,:] = image_predicted[i, :, H_min:H_max+ 1, W_min:W_max+1]
                image_combined_pred_target[1,:,:,:] = image_target[i, :, H_min:H_max+ 1, W_min:W_max+1]
                image_combined_pred_target = image_combined_pred_target.type(torch.FloatTensor).to(self.device)

                loss_localNCC = self.run(image_combined_pred_target)
                loss_localNCC_total +=loss_localNCC
                ROI.append(ROI_case)
        
        if self.reduction == 'mean':
            loss_localNCC_total = loss_localNCC_total/image_predicted.shape[0]
            loss_localNCC_total = loss_localNCC_total.type(torch.FloatTensor)
            loss_localNCC_total = loss_localNCC_total.to(self.device)
            # print('batch size: {}'.format(image_predicted.shape[0]))
            
        if self.reduction == 'sum':
            # TODO: need to implement this
            loss_localNCC_total = loss_localNCC_total.type(torch.FloatTensor)
            loss_localNCC_total = loss_localNCC_total.to(self.device)
        
        return -loss_localNCC_total, ROI

    def run(self, image_combined_pred_target):
        """Calculate the local normalized cross correlation (default: dim =2)
        image_combined_pred_target: [B, C, H, W] """
        """
        Local squared zero-normalized cross-correlation.

        Denote y_true as t and y_pred as p. Consider a window having n elements.
        Each position in the window corresponds a weight w_i for i=1:n.

        Define the discrete expectation in the window E[t] as

            E[t] = sum_i(w_i * t_i) / sum_i(w_i)

        Similarly, the discrete variance in the window V[t] is

            V[t] = E[t**2] - E[t] ** 2

        The local squared zero-normalized cross-correlation is therefore

            E[ (t-E[t]) * (p-E[p]) ] ** 2 / V[t] / V[p]

        where the expectation in numerator is

            E[ (t-E[t]) * (p-E[p]) ] = E[t * p] - E[t] * E[p]

        Different kernel corresponds to different weights.

        Reference:

            - Zero-normalized cross-correlation (ZNCC):
                https://en.wikipedia.org/wiki/Cross-correlation
            - Code: https://github.com/voxelmorph/voxelmorph/blob/legacy/src/losses.py
        """
        
        pred = image_combined_pred_target[0, :, :, :].unsqueeze(0)
        target = image_combined_pred_target[1, :, :, :].unsqueeze(0)
        pred2 = pred*pred 
        target2 = target*target
        pred_target = pred*target

        # compute the sum filter
        sum_filter = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
        sum_filter = sum_filter.to(self.device)
        pred_sum = F.conv2d(pred, sum_filter, stride= self.stride, padding = self.padding)
        target_sum = F.conv2d(target, sum_filter, stride = self.stride, padding = self.padding)
        pred2_sum = F.conv2d(pred2, sum_filter, stride = self.stride, padding = self.padding)
        target2_sum = F.conv2d(target2, sum_filter, stride = self.stride, padding = self.padding)
        pred_target_sum = F.conv2d(pred_target, sum_filter, stride = self.stride, padding = self.padding)
        # print("pred2_sum", pred2_sum)
        # average over kernel
        
        win_size = self.kernel_size[0] * self.kernel_size[1]
        u_pred = pred_sum/win_size
        u_target = target_sum/win_size

        cross = pred_target_sum - u_pred*target_sum
        target_var = torch.max(target2_sum - u_target * target_sum, torch.as_tensor(self.eps, dtype=target2_sum.dtype, device=target2_sum.device))
        pred_var = torch.max(pred2_sum - u_pred * pred_sum, torch.as_tensor(self.eps, dtype=pred2_sum.dtype, device=pred2_sum.device))
        
        loss_localNCC = cross*cross / (target_var*pred_var)

        # filter out the blank areas
        
        pred2_sum_flatten = torch.flatten(pred2_sum)
        nonzero_mask = torch.zeros(pred2_sum_flatten.shape)
        nonzero_mask[pred2_sum_flatten > self.win_eps] = 1
        nonzero_mask = nonzero_mask.to(self.device)
        # print("self.win_eps is: ", self.win_eps)
        
        nonzero_indices = torch.nonzero(pred2_sum_flatten*nonzero_mask, as_tuple = False)
        loss_localNCC_flatten = torch.flatten(loss_localNCC)
        loss_localNCC_mean = torch.mean(loss_localNCC_flatten[nonzero_indices])
        # print("valid windows: {}/{}".format(nonzero_indices.shape[0],pred2_sum_flatten.shape))
        # loss_localNCC_mean = torch.mean(loss_localNCC_flatten)
        # loss_localNCC_mean = torch.mean(loss_localNCC, dim =(0,1,2,3))   
        return loss_localNCC_mean

class GradientNCC_new:
    def __init__(self, device = "cuda:0", kernel_size =(51, 51), stride = (1, 1), sampling = 'regular', padding = 'same', reduction = 'mean', cropping = True, eps = 1e-5, win_eps = 0.98):
        
        self.kernel_size = kernel_size
        # if self.kernel_size == 'none':
        #     self.kernel_size = (21, 21)
        self.sampling = sampling
        self.reduction = reduction
        self.stride = stride
        self.padding = padding
        self.eps = eps
        self.cropping = cropping
        self.win_eps = win_eps
        self.device = device

    def __call__(self, image_predicted, image_target):
        """Preprocess the images, to avoid the sub-optimal problem"""
        """image_predicted: tensor, N*1*H*W; image_target: tensor, N*1*H*W"""
        # preprocessing the image (define the ROI using the intersection of image_predicted and image_target)
        loss_localNCC_total = 0
        ROI = []
        if self.cropping:

            # ROI cropping
            image_predicted_mask = torch.zeros(image_predicted.shape)
            image_predicted_mask[image_predicted > 0.0] = 1
            image_predicted_mask = image_predicted_mask.to(self.device)    
            image_target_mask = torch.zeros(image_target.shape)
            image_target_mask[image_target > 0.0] = 1
            image_target_mask = image_target_mask.to(self.device)
            image_ROI = image_predicted_mask * image_target_mask # should be leaf = false, requires_grad = true
            
            # valid irregular ROI cropping
            sum_filter = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
            sum_filter = sum_filter.to(device=self.device)
            image_ROI_sum = F.conv2d(image_ROI, sum_filter, stride= 1, padding = "same")
            image_ROI_valid_mask = torch.ones(image_ROI_sum.shape)
            image_ROI_valid_mask = image_ROI_valid_mask.to(device = self.device)
            threshold = int(self.kernel_size[0]*self.kernel_size[1] * self.win_eps)
            image_ROI_valid_mask = image_ROI_valid_mask * (image_ROI_sum > threshold) # requires_grad = False
            # print("image_ROI_valid_mask requires_grad: {}".format(image_ROI_valid_mask.requires_grad))

            for i in range(image_predicted.shape[0]):
                indices = torch.nonzero(image_ROI[i, 0, :, :], as_tuple = True)
                if indices[0].numel() == 0:
                    H_min, H_max = int(image_predicted.shape[2]*0.5 - 0.6*self.kernel_size[0]), int(image_predicted.shape[2]*0.5 + 0.6*self.kernel_size[0])
                    W_min, W_max = int(image_predicted.shape[3]*0.5 - 0.6*self.kernel_size[1]), int(image_predicted.shape[3]*0.5 + 0.6*self.kernel_size[1])
                else:
                    H_min, H_max = torch.min(indices[0]), torch.max(indices[0])
                    W_min, W_max = torch.min(indices[1]), torch.max(indices[1])
                    # print("(1): H_min: {}, H_max: {}, W_min: {}, W_max: {}".format(H_min, H_max, W_min, W_max))
                    if (H_max-H_min + 1) < self.kernel_size[0]:
                        H_min_ = torch.min(H_min, torch.floor(((H_max+H_min)*0.5 - self.kernel_size[0]*0.5 -1)).type(torch.int16))
                        H_max_ = torch.max(H_max, torch.ceil(((H_max+H_min)*0.5 + self.kernel_size[0]*0.5)).type(torch.int16))
                        if H_min_ < 1:
                            H_max = H_max_ - H_min_ +1
                            H_min = 1
                        elif H_max_ > image_predicted.shape[2]-1:
                            H_min = H_min_ - (H_max_ - image_predicted.shape[2]).type(torch.int16)
                            H_max = image_predicted.shape[2]-1
                        else:
                            H_max = H_max_
                            H_min = H_min_
                    if (W_max-W_min + 1) < self.kernel_size[1]:
                        W_min_ = torch.min(W_min, torch.floor(((W_max+W_min)*0.5- self.kernel_size[1]*0.5 -1)).type(torch.int16))
                        W_max_ = torch.max(W_max, torch.ceil(((W_max+W_min)*0.5+ self.kernel_size[1]*0.5)).type(torch.int16))
                        if W_min_ < 1:
                            W_max = W_max_ -W_min_ + 1
                            W_min = 1
                        elif W_max_ > image_predicted.shape[3]-1:
                            W_min = W_min_ - (W_max_ - image_predicted.shape[3]).type(torch.int16)
                            W_max = image_predicted.shape[3]-1
                        else:
                            W_min = W_min_
                            W_max = W_max_
                ROI_case =  {"H_min": H_min, "H_max": H_max, "W_min": W_min, "W_max": W_max}
                # print(image_combined_pred_target.shape)
                # print("H_min: {}, H_max: {}, W_min: {}, W_max: {}".format(H_min, H_max, W_min, W_max))
                

                # image_pred= image_predicted[i, :, :, :]
                # image_tar = image_target[i, :, :, :]
                # loss_localNCC = self.run(image_pred, image_tar, image_ROI_valid_mask)
                
                image_pred = image_predicted[i, :, H_min:H_max+ 1, W_min:W_max+1] # requires_grad = True
                image_tar = image_target[i, :, H_min:H_max+ 1, W_min:W_max+1]
                image_ROI_valid_mask_ = image_ROI_valid_mask[i, :, H_min:H_max+ 1, W_min:W_max+1].unsqueeze(0) # requires_grad = False
                # print("image_pred requires_grad: {}".format(image_pred.requires_grad)) 
                # print("image_tar requires_grad: {}".format(image_tar.requires_grad)) 
                # print("image_ROI_valid_mask_ requires_grad: {}".format(image_ROI_valid_mask_.requires_grad)) 
                # print("image_ROI_valid_mask_ shape: {}".format(image_ROI_valid_mask_.shape))
                # loss_localNCC = self.run(image_pred, image_tar, image_ROI_valid_mask_)
                loss_localNCC = self.run_gradientNCC(image_pred, image_tar, image_ROI_valid_mask_)
                # print("loss_localNCC (device): ", loss_localNCC.device)
                loss_localNCC_total +=loss_localNCC
                ROI.append(ROI_case)


                # roi_valid_mask_np = torch.Tensor.numpy(image_ROI_valid_mask.detach().cpu())
                # roi_mask_np = torch.Tensor.numpy(image_ROI.detach().cpu())
                # plt.figure("visualize", (12,8))
                # plt.subplot(1,2,1)
                # plt.title("original image")
                # # plt.imshow(np.transpose(image_array_frame[:,:,int(image_array_frame.shape[2]/2)]), cmap="gray")
                # plt.imshow(roi_mask_np[0, 0,:,:], cmap="gray")
                # plt.gca().add_patch(Rectangle((float(ROI[0]['W_min']), float(ROI[0]['H_min'])), float(ROI[0]['W_max'] - ROI[0]['W_min']+1), float(ROI[0]['H_max'] - ROI[0]['H_min']+1), edgecolor = 'red', facecolor='none'))

                # plt.subplot(1,2,2)
                # plt.imshow(roi_valid_mask_np[0, 0,:,:], cmap="gray")
                # plt.title("ROIed image")
                # plt.gca().add_patch(Rectangle((float(ROI[0]['W_min']), float(ROI[0]['H_min'])), float(ROI[0]['W_max'] - ROI[0]['W_min']+1), float(ROI[0]['H_max'] - ROI[0]['H_min']+1), edgecolor = 'red', facecolor='none'))

        else:
            for i in range(image_predicted.shape[0]):

                indices = torch.nonzero(image_ROI_valid_mask[i, 0, :, :], as_tuple = True)
                # print("indices: ", indices)
                if indices[0].numel() == 0:
                    H_min, H_max = int(image_predicted.shape[2]*0.5 - 0.6*self.kernel_size[0]), int(image_predicted.shape[2]*0.5 + 0.6*self.kernel_size[0])
                    W_min, W_max = int(image_predicted.shape[3]*0.5 - 0.6*self.kernel_size[1]), int(image_predicted.shape[3]*0.5 + 0.6*self.kernel_size[1])
                else:
                    H_min, H_max = torch.min(indices[0]), torch.max(indices[0])
                    W_min, W_max = torch.min(indices[1]), torch.max(indices[1])
                    # print("(1): H_min: {}, H_max: {}, W_min: {}, W_max: {}".format(H_min, H_max, W_min, W_max))
                    

                    if (H_max-H_min + 1) < self.kernel_size[0]:
                        H_min_ = torch.min(H_min, torch.floor(((H_max+H_min)*0.5 - self.kernel_size[0]*0.5 -1)).type(torch.int16))
                        H_max_ = torch.max(H_max, torch.ceil(((H_max+H_min)*0.5 + self.kernel_size[0]*0.5)).type(torch.int16))
                        # print("H_min_: {}".format(H_min_))
                        # print("H_max_: {}".format(H_max_))
                        if H_min_ < 1:
                            H_max = H_max_ - H_min_ +1
                            H_min = 1
                        elif H_max_ > image_predicted.shape[2]-1:
                            H_min = H_min_ - (H_max_ - image_predicted.shape[2]).type(torch.int16)
                            H_max = image_predicted.shape[2]-1
                        else:
                            H_max = H_max_
                            H_min = H_min_
                        # print("H_min_new: {}".format(H_min))
                        # print("H_max_new: {}".format(H_max))
                    if (W_max-W_min + 1) < self.kernel_size[1]:
                        W_min_ = torch.min(W_min, torch.floor(((W_max+W_min)*0.5- self.kernel_size[1]*0.5 -1)).type(torch.int16))
                        W_max_ = torch.max(W_max, torch.ceil(((W_max+W_min)*0.5+ self.kernel_size[1]*0.5)).type(torch.int16))
                        if W_min_ < 1:
                            W_max = W_max_ -W_min_ + 1
                            W_min = 1
                        elif W_max_ > image_predicted.shape[3]-1:
                            W_min = W_min_ - (W_max_ - image_predicted.shape[3]).type(torch.int16)
                            W_max = image_predicted.shape[3]-1
                        else:
                            W_min = W_min_
                            W_max = W_max_
                            

                    
                ROI_case =  {"H_min": H_min, "H_max": H_max, "W_min": W_min, "W_max": W_max}
                
                ROI.append(ROI_case)
                # print("non-cropping")
        
        if self.reduction == 'mean':
            loss_localNCC_total = loss_localNCC_total/image_predicted.shape[0]
            # print("loss_localNCC_total requires_grad: {}".format(loss_localNCC_total.requires_grad))
            # print(loss_localNCC_total)
            loss_localNCC_total = loss_localNCC_total.type(torch.FloatTensor)
            loss_localNCC_total = loss_localNCC_total.to(self.device)
            # print('batch size: {}'.format(image_predicted.shape[0]))
            
        if self.reduction == 'sum':
            # TODO: need to implement this
            loss_localNCC_total = loss_localNCC_total.type(torch.FloatTensor)
            loss_localNCC_total = loss_localNCC_total.to(self.device)
        
        return -loss_localNCC_total, ROI

    def run_localNCC(self, image_pred, image_tar, image_ROI_valid_mask):
        """Calculate the local normalized cross correlation (default: dim =2)
        image_combined_pred_target: [B, C, H, W] """
        """
        Local squared zero-normalized cross-correlation.
        Denote y_true as t and y_pred as p. Consider a window having n elements.
        Each position in the window corresponds a weight w_i for i=1:n.

        Define the discrete expectation in the window E[t] as
            E[t] = sum_i(w_i * t_i) / sum_i(w_i)

        Similarly, the discrete variance in the window V[t] is
            V[t] = E[t**2] - E[t] ** 2

        The local squared zero-normalized cross-correlation is therefore
            E[ (t-E[t]) * (p-E[p]) ] ** 2 / V[t] / V[p]

        where the expectation in numerator is
            E[ (t-E[t]) * (p-E[p]) ] = E[t * p] - E[t] * E[p]

        Different kernel corresponds to different weights.
        Reference:
            - Zero-normalized cross-correlation (ZNCC):
                https://en.wikipedia.org/wiki/Cross-correlation
            - Code: https://github.com/voxelmorph/voxelmorph/blob/legacy/src/losses.py
        """
        
        pred = image_pred
        target = image_tar
        pred2 = pred*pred 
        target2 = target*target
        pred_target = pred*target

        # compute the sum filter
        sum_filter = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
        sum_filter = sum_filter.to(self.device)
        one_filter = torch.zeros(1, 1, self.kernel_size[0], self.kernel_size[1])
        one_filter[:,:, int((self.kernel_size[0]-1)*0.5), int((self.kernel_size[1]-1)*0.5)] = 1
        one_filter.type(torch.FloatTensor)
        one_filter = one_filter.to(self.device)

        pred_sum = F.conv2d(pred, sum_filter, stride= self.stride, padding = self.padding)
        target_sum = F.conv2d(target, sum_filter, stride = self.stride, padding = self.padding)
        pred_target_sum = F.conv2d(pred_target, sum_filter, stride = self.stride, padding = self.padding)

        # print("pred_sum shape: {}".format(pred_sum.shape))
        # print("target_sum shape: {}".format(target_sum.shape))
        # print("pred_sum requires_grad: {}".format(pred_sum.requires_grad)) 
        # print("pred_target_sum requires_grad: {}".format(pred_target_sum.requires_grad)) 

        pred2_sum = F.conv2d(pred2, sum_filter,stride = self.stride, padding = self.padding)
        target2_sum = F.conv2d(target2, sum_filter,stride = self.stride, padding = self.padding)
        # print("pred2_sum shape: {}".format(pred2_sum.shape))
        # print("target2_sum shape: {}".format(target2_sum.shape))
        image_ROI_valid_mask_filtered = F.conv2d(image_ROI_valid_mask, one_filter, stride=self.stride, padding=self.padding) # this is one_filter is to use the "stride" function
        # print("image_ROI_valid_mask_filtered shape: {}".format(image_ROI_valid_mask_filtered.shape))
        # print("image_ROI_valid_mask shape: {}".format(image_ROI_valid_mask.shape))
        # print("pred2_sum", pred2_sum)
        # average over kernel
        
        win_size = self.kernel_size[0] * self.kernel_size[1]
        u_pred = pred_sum/win_size
        u_target = target_sum/win_size

        cross = pred_target_sum - u_pred*target_sum
        target_var = torch.max(target2_sum - u_target * target_sum, torch.as_tensor(self.eps, dtype=target2_sum.dtype, device=target2_sum.device))
        pred_var = torch.max(pred2_sum - u_pred * pred_sum, torch.as_tensor(self.eps, dtype=pred2_sum.dtype, device=pred2_sum.device))
        
        # loss_localNCC = cross*cross / (target_var*pred_var)
        loss_localNCC = cross / (torch.sqrt(target_var*pred_var))
        # loss_localNCC = torch.sqrt(loss_localNCC)
        # print("loss_localNCC requires_grad: {}".format(loss_localNCC.requires_grad))
        # print("target_var variance: {}".format(target_var.shape))
        
        """test"""
        # target_var_np = torch.Tensor.numpy(target_var.detach().cpu())

        # target_var_new = target_var*image_ROI_valid_mask_filtered
        # target_var_new_np = torch.Tensor.numpy(target_var_new.detach().cpu())

        # plt.figure("visualize", (8,4))
        # plt.subplot(1,2,1)
        # plt.title("original image")
        # # plt.imshow(np.transpose(image_array_frame[:,:,int(image_array_frame.shape[2]/2)]), cmap="gray")
        # plt.imshow(target_var_np[0, 0,:,:], cmap="hot")
        # plt.colorbar()
        # # print(target_var_np[0, 0,:,:])
        # plt.subplot(1,2,2)
        # plt.title("refined image")
        # # plt.imshow(np.transpose(image_array_frame[:,:,int(image_array_frame.shape[2]/2)]), cmap="gray")
        # plt.imshow(target_var_new_np[0, 0,:,:], cmap="hot")
        # plt.colorbar()

  
        # print("loss_localNCC dtype: {}".format(loss_localNCC.dtype))
        # # filter out the blank areas
        # image_ROI_valid_mask_flatten = torch.flatten(image_ROI_valid_mask).to(device=loss_localNCC.device)
        image_ROI_valid_mask_flatten = torch.flatten(image_ROI_valid_mask_filtered).to(device=loss_localNCC.device)
        nonzero_indices = torch.nonzero(image_ROI_valid_mask_flatten, as_tuple = False)
        # print("nonzero_indices shape: {}".format(nonzero_indices.shape))
        loss_localNCC_flatten = torch.flatten(loss_localNCC)
        # print("loss_localNCC_flatten shape: {}".format(loss_localNCC_flatten.shape))

        if nonzero_indices.numel() == 0:
            # nonzero_indices = torch.Tensor([int(image_ROI_valid_mask_flatten.shape[0]*0.5)]).type(torch.int)
            
            # loss_localNCC_mean = torch.mean(loss_localNCC_flatten[nonzero_indices]) / (int(image_ROI_valid_mask_flatten.shape[0]))
            loss_localNCC_mean = torch.mean(loss_localNCC_flatten[int(image_ROI_valid_mask_flatten.shape[0]*0.5)]) / (int(image_ROI_valid_mask_flatten.shape[0]))
            # print("loss_localNCC_mean(0) requires_grad: {}".format(loss_localNCC_mean.requires_grad))
        else:
            # loss_localNCC_mean = torch.mean(loss_localNCC_flatten)
            loss_localNCC_mean = torch.mean(loss_localNCC_flatten[nonzero_indices])
            # print("loss_localNCC_mean requires_grad: {}".format(loss_localNCC_mean.requires_grad))

        return loss_localNCC_mean
    
    def run_gradientNCC(self, image_pred, image_tar, image_ROI_valid_mask):
        """Calculate the gradient local normalized cross correlation (default: dim =2)
        image_combined_pred_target: [B, C, H, W] """
        image_pred = image_pred.unsqueeze(0)
        image_tar = image_tar.unsqueeze(0)

        # with torch.no_grad():
        sobel_kernel_X = torch.Tensor([[[[1 ,0, -1],[2, 0 ,-2], [1, 0 ,-1]]]])
        sobel_kernel_X = sobel_kernel_X.to(self.device)
        sobel_kernel_Y = torch.Tensor([[[[1, 2, 1],[0, 0, 0], [-1, -2 ,-1]]]])
        sobel_kernel_Y = sobel_kernel_Y.to(self.device)

        gaussian_kernel = torch.tensor([[[[0.0029, 0.0131, 0.0215, 0.0131, 0.0029],
          [0.0131, 0.0585, 0.0965, 0.0585, 0.0131],
          [0.0215, 0.0965, 0.1592, 0.0965, 0.0215],
          [0.0131, 0.0585, 0.0965, 0.0585, 0.0131],
          [0.0029, 0.0131, 0.0215, 0.0131, 0.0029]]]])
        # gaussian_kernel = torch.tensor([[[[0.0146, 0.0213, 0.0241, 0.0213, 0.0146],
        #   [0.0213, 0.0310, 0.0351, 0.0310, 0.0213],
        #   [0.0241, 0.0351, 0.0398, 0.0351, 0.0241],
        #   [0.0213, 0.0310, 0.0351, 0.0310, 0.0213],
        #   [0.0146, 0.0213, 0.0241, 0.0213, 0.0146]]]])
        # gaussian_kernel = torch.ones((1, 1, 3, 3))/9
        gaussian_kernel = gaussian_kernel.to(self.device)
        image_pred = F.conv2d(image_pred, gaussian_kernel, stride=1, padding ="same") # requires_grad= True
        image_tar = F.conv2d(image_tar, gaussian_kernel, stride=1, padding ="same")# requires_grad= False

        image_pred_x = F.conv2d(image_pred, sobel_kernel_X, stride=1, padding =1) # requires_grad= True
        image_pred_y = F.conv2d(image_pred, sobel_kernel_Y, stride=1, padding =1)
        image_tar_x = F.conv2d(image_tar, sobel_kernel_X, stride=1, padding =1)# requires_grad= False
        image_tar_y = F.conv2d(image_tar, sobel_kernel_Y, stride=1, padding =1)   
        
        # image_pred_x = F.conv2d(image_pred_x, gaussian_kernel, stride=1, padding ="same") # requires_grad= True
        # image_pred_y = F.conv2d(image_pred_y, gaussian_kernel, stride=1, padding ="same")# requires_grad= False
        # image_tar_x = F.conv2d(image_tar_x, gaussian_kernel, stride=1, padding ="same") # requires_grad= True
        # image_tar_y = F.conv2d(image_tar_y, gaussian_kernel, stride=1, padding ="same")# requires_grad= False
        # print("blurring after")
        # print("image_pred_x requires_grad: {}".format(image_pred_x.requires_grad))
        # print("image_tar_x requires_grad: {}".format(image_tar_x.requires_grad))

        # image_pred_x_clone = torch.abs(image_pred_x)*image_ROI_valid_mask
        # image_tar_x_clone = torch.abs(image_tar_x)*image_ROI_valid_mask
        # image_pred_y_clone = torch.abs(image_pred_y)*image_ROI_valid_mask
        # image_tar_y_clone = torch.abs(image_tar_y)*image_ROI_valid_mask
        # image_pred_x_np = torch.Tensor.numpy(image_pred_x_clone.detach().cpu())
        # image_tar_x_np = torch.Tensor.numpy(image_tar_x_clone.detach().cpu())
        # image_pred_y_np = torch.Tensor.numpy(image_pred_y_clone.detach().cpu())
        # image_tar_y_np = torch.Tensor.numpy(image_tar_y_clone.detach().cpu())
        # image_pre_xy_np = image_pred_x_np + image_pred_y_np
        # image_tar_xy_np = image_tar_x_np + image_tar_y_np
        # plt.figure(figsize=(16, 6))
        # plt.subplot(2,3,1)
        # plt.imshow(image_pred_x_np[0, 0, :, :], cmap = "hot")
        # plt.title("image pred (x)")
        # plt.colorbar()

        # plt.subplot(2,3,2)
        # plt.imshow(image_pred_y_np[0, 0, :, :], cmap = "hot")
        # plt.title("image pred(x)")
        # plt.colorbar()

        # plt.subplot(2,3,3)
        # plt.imshow(image_pre_xy_np[0, 0, :, :], cmap = "hot")
        # plt.title("image pred (x+y)")
        # plt.colorbar()

        # plt.subplot(2,3,4)
        # plt.imshow(image_tar_x_np[0, 0, :, :], cmap = "hot")
        # plt.title("image target(x)")
        # plt.colorbar()

        # plt.subplot(2,3,5)
        # plt.imshow(image_tar_y_np[0, 0, :, :], cmap = "hot")
        # plt.title("image target(y)")
        # plt.colorbar()

        # plt.subplot(2,3,6)
        # plt.imshow(image_tar_y_np[0, 0, :, :], cmap = "hot")
        # plt.title("image target(x + y)")
        # plt.colorbar()

        # gradient_global_ncc = self.cal_ncc(torch.abs(image_pred_x), torch.abs(image_tar_x), image_ROI_valid_mask, self.eps) + self.cal_ncc(torch.abs(image_pred_y), torch.abs(image_tar_y), image_ROI_valid_mask, self.eps)
        # gradient_local_ncc_x = self.run_localNCC(torch.abs(image_pred_x), torch.abs(image_tar_x), image_ROI_valid_mask) 
        # gradient_local_ncc_y = self.run_localNCC(torch.abs(image_pred_y), torch.abs(image_tar_y), image_ROI_valid_mask)
        gradient_local_ncc_xy = self.run_localNCC(torch.abs(image_pred_x) + torch.abs(image_pred_y), torch.abs(image_tar_x) + torch.abs(image_tar_y), image_ROI_valid_mask) 

        # gradient_local_ncc_x = self.run_localNCC(image_pred_x, image_tar_x, image_ROI_valid_mask) 
        # gradient_local_ncc_y = self.run_localNCC(image_pred_y, image_tar_y, image_ROI_valid_mask)
        # # print("gradient_local_ncc_x requires_grad: {}".format(gradient_local_ncc_x.requires_grad))
        # # print("gradient_local_ncc_y requires_grad: {}".format(gradient_local_ncc_y.requires_grad))
        # print("gradient_local_ncc_x: {}, gradient_local_ncc_y: {}".format(gradient_local_ncc_x, gradient_local_ncc_y))
        # return gradient_local_ncc_x + gradient_local_ncc_y
        # print("gradient_local_ncc_xy: {}".format(gradient_local_ncc_xy))
        return gradient_local_ncc_xy
    
    def run_globalNCC(self, image_pred, image_tar, image_ROI_valid_mask):
        # compute local sums via convolution(this is global ncc) 
        cross = (image_pred - torch.mean(image_pred)) * (image_tar - torch.mean(image_tar))
        image_pred_var = (image_pred - torch.mean(image_pred)) * (image_pred - torch.mean(image_pred))
        image_tar_var = (image_tar - torch.mean(image_tar)) * (image_tar - torch.mean(image_tar))
        # print("image_tar_var requires_grad: {}".format(image_tar_var.requires_grad))
        gradient_globalNCC = cross / torch.sqrt(image_pred_var*image_tar_var + self.eps)
        # print("gradient_globalNCC requires_grad: {}".format(gradient_globalNCC.requires_grad))
        image_ROI_valid_mask_flatten = torch.flatten(image_ROI_valid_mask).to(device=self.device)
        nonzero_indices = torch.nonzero(image_ROI_valid_mask_flatten, as_tuple = False)
        
        loss_gradient_globalNCC_flatten = torch.flatten(gradient_globalNCC)
        if nonzero_indices.numel() == 0:
            nonzero_indices = torch.Tensor([int(image_ROI_valid_mask_flatten.shape[0]*0.5)]).type(torch.int)
            loss_gradient_globalNCC_mean = torch.mean(loss_gradient_globalNCC_flatten[nonzero_indices])
            # print("loss_gradient_globalNCC_mean XXXXX requires_grad: {}".format(loss_gradient_globalNCC_mean.requires_grad))
        else:
            loss_gradient_globalNCC_mean = torch.mean(loss_gradient_globalNCC_flatten[nonzero_indices])
        # print("gradient_globalNCC requires_grad: {}".format(gradient_globalNCC.requires_grad))
        # test = torch.mean(cc)
        return loss_gradient_globalNCC_mean

class gradientNCC:
    def __init__(self, device = "cuda:0",  kernel_size =(101, 101), eps = 1e-10, cropping = True, boundary_shift = 3, fig = None):

        self.device = device
        self.eps = eps
        self.kernel_size = kernel_size
        self.cropping = cropping
        self.boundary_shift = boundary_shift
        self.fig = fig
    def __call__(self, image_predicted, image_target):
        """Preprocess the images, to avoid the sub-optimal problem"""
        """image_predicted: tensor, N*1*H*W; image_target: tensor, N*1*H*W"""
        ROI = []
        loss_gradientNCC_total = 0
        if self.cropping:

            image_predicted_mask = torch.zeros(image_predicted.shape)
            image_predicted_mask[image_predicted > 0.0] = 1
            image_predicted_mask = image_predicted_mask.to(self.device)    
            image_target_mask = torch.zeros(image_target.shape)
            image_target_mask[image_target > 0.0] = 1
            image_target_mask = image_target_mask.to(self.device)
            image_ROI = image_predicted_mask * image_target_mask # should be leaf = false, requires_grad = true
            # print("image_ROI shape: ", image_ROI.shape)
            
            for i in range(image_predicted.shape[0]):
                indices = torch.nonzero(image_ROI[i, 0, :, :], as_tuple = True)
                # print("indices: ", indices)
                if indices[0].numel() == 0:
                    H_min, H_max = 0, image_predicted.shape[2]-1
                    W_min, W_max = 0, image_predicted.shape[3]-1
                else:
                    H_min, H_max = torch.min(indices[0])+ self.boundary_shift, torch.max(indices[0]) -self.boundary_shift
                    W_min, W_max = torch.min(indices[1])+ self.boundary_shift, torch.max(indices[1])- self.boundary_shift
                    if (H_max-H_min) < 0 or (W_max-W_min) < 0:
                        H_min, H_max = torch.min(indices[0]), torch.max(indices[0]) 
                        W_min, W_max = torch.min(indices[1]), torch.max(indices[1])
                    
                ROI_case =  {"H_min": H_min, "H_max": H_max, "W_min": W_min, "W_max": W_max}
                # print(ROI_case)
                # ROIed
                # image_combined_pred_target = torch.zeros(2, 1, H_max-H_min+1, W_max-W_min+1)
                
                # print(image_combined_pred_target.shape)
                # print("H_min: {}, H_max: {}, W_min: {}, W_max: {}".format(H_min, H_max, W_min, W_max))
                image_pred = image_predicted[i, :, H_min:H_max+ 1, W_min:W_max+1].type(torch.FloatTensor)
                image_tar = image_target[i, :, H_min:H_max+ 1, W_min:W_max+1].type(torch.FloatTensor)
                image_pred = image_pred.to(self.device)
                image_tar = image_tar.to(self.device)
                
                # print("image_combined_pred_target (require_grad): {}".format(image_combined_pred_target.requires_grad))
                loss_gradientNCC = self.run(image_pred, image_tar)
                # print("loss_localNCC (device): ", loss_localNCC.device)
                loss_gradientNCC_total +=loss_gradientNCC
                ROI.append(ROI_case)

        else:
            for i in range(image_predicted.shape[0]):

                # image_combined_pred_target = torch.zeros(2, 1, image_predicted.shape[2], image_predicted.shape[3])
                H_min, H_max = 0, image_predicted.shape[2]-1
                W_min, W_max = 0, image_predicted.shape[3]-1
                ROI_case =  {"H_min": H_min, "H_max": H_max, "W_min": W_min, "W_max": W_max}
                
                # print(image_combined_pred_target.shape)
                # print("H_min: {}, H_max: {}, W_min: {}, W_max: {}".format(H_min, H_max, W_min, W_max))
                image_pred = image_predicted[i, :, H_min:H_max+ 1, W_min:W_max+1].type(torch.FloatTensor)
                image_pred = image_pred.to(self.device)
                image_tar = image_target[i, :, H_min:H_max+ 1, W_min:W_max+1].type(torch.FloatTensor)
                image_tar = image_tar.to(self.device)
                
                # print("image_combined_pred_target (require_grad): {}".format(image_combined_pred_target.requires_grad))
                loss_gradientNCC = self.run(image_pred, image_tar)

                
                loss_gradientNCC_total +=loss_gradientNCC
                ROI.append(ROI_case)

        loss_gradientNCC_total = loss_gradientNCC_total/image_predicted.shape[0]
        loss_gradientNCC_total = loss_gradientNCC_total.type(torch.FloatTensor)
        loss_gradientNCC_total = loss_gradientNCC_total.to(self.device)

        return loss_gradientNCC_total, ROI

        
    def run(self, image_pred, image_tar):
        """Calculate the gradient local normalized cross correlation (default: dim =2)
        image_combined_pred_target: [B, C, H, W] """
        image_pred = image_pred.unsqueeze(0)
        image_tar = image_tar.unsqueeze(0)

        with torch.no_grad():
            kernel_X = torch.Tensor([[[[1 ,0, -1],[2, 0 ,-2], [1, 0 ,-1]]]])
            kernel_X = torch.nn.Parameter( kernel_X, requires_grad = False )
            kernel_Y = torch.Tensor([[[[1, 2, 1],[0, 0, 0], [-1, -2 ,-1]]]])
            kernel_Y = torch.nn.Parameter( kernel_Y, requires_grad = False )
            SobelX = nn.Conv2d( 1, 1, 3, 1, 1, bias=False)
            SobelX.weight = kernel_X
            SobelY = nn.Conv2d( 1, 1, 3, 1, 1, bias=False)
            SobelY.weight = kernel_Y

            SobelX = SobelX.to(self.device)
            SobelY = SobelY.to(self.device)

        image_pred_x = SobelX(image_pred)
        image_pred_y = SobelY(image_pred)
        image_tar_x = SobelX(image_tar)
        image_tar_y = SobelY(image_tar)
        # gradient_ncc = 1-0.5*self.cal_ncc(image_pred_x, image_tar_x, self.eps)-0.5*self.cal_ncc(image_pred_y, image_tar_y, self.eps)
        gradient_ncc = self.cal_ncc(image_pred_x, image_tar_x, self.eps) + self.cal_ncc(image_pred_y, image_tar_y, self.eps)
        image_pred_x_clone = image_pred_x.clone()
        image_tar_x_clone = image_tar_x.clone()
        image_pred_y_clone = image_pred_y.clone()
        image_tar_y_clone = image_tar_y.clone()
        image_pred_x_np = torch.Tensor.numpy(image_pred_x_clone.detach().cpu())
        image_tar_x_np = torch.Tensor.numpy(image_tar_x_clone.detach().cpu())
        image_pred_y_np = torch.Tensor.numpy(image_pred_y_clone.detach().cpu())
        image_tar_y_np = torch.Tensor.numpy(image_tar_y_clone.detach().cpu())

        # fig = plt.figure(figsize=(16, 9))
        # ax1 = self.fig.add_subplot(221)

        # ax1.imshow(image_pred_x_np[0, 0, :, :], cmap = "gray")
        # ax1.set_title("image pred (x)")
        # ax2 = self.fig.add_subplot(222)
        # ax2.imshow(image_tar_x_np[0, 0, :, :], cmap = "gray")
        # ax2.set_title("image target(x)")
        # ax1 = self.fig.add_subplot(223)
        # ax1.imshow(image_pred_y_np[0, 0, :, :], cmap = "gray")
        # ax1.set_title("image pred (y)")
        # ax2 = self.fig.add_subplot(224)
        # ax2.imshow(image_tar_y_np[0, 0, :, :], cmap = "gray")
        # ax2.set_title("image target(y)")
        # plt.show(block=False)
        # plt.pause(0.5)
        # plt.clf()
        
        return  gradient_ncc
    
    def cal_ncc(self, image_pred, image_tar, eps):
        # compute local sums via convolution(this is global ncc) 
        cross = (image_pred - torch.mean(image_pred)) * (image_tar - torch.mean(image_tar))
        image_pred_var = (image_pred - torch.mean(image_pred)) * (image_pred - torch.mean(image_pred))
        image_tar_var = (image_tar - torch.mean(image_tar)) * (image_tar - torch.mean(image_tar))

        cc = torch.sum(cross) / torch.sum(torch.sqrt(image_pred_var*image_tar_var + eps))

        # test = torch.mean(cc)
        return -torch.mean(cc)

class LC2:
    def __init__(self, radiuses=(3,5,7)):
        self.radiuses = radiuses
        self.f = torch.zeros(3, 1, 3, 3, 3)
        self.f[0, 0, 1, 1, 0] = 1
        self.f[0, 0, 1, 1, 2] = -1
        self.f[1, 0, 1, 0, 1] = 1
        self.f[1, 0, 1, 2, 1] = -1
        self.f[2, 0, 0, 1, 1] = 1
        self.f[2, 0, 2, 1, 1] = -1

    def __call__(self, us, mr):
        s = self.run(us, mr, self.radiuses[0])
        for r in self.radiuses[1:]:
            s += self.run(us, mr, r)
        return s / len(self.radiuses)

    def run(self, us, mr, radius=9, alpha=1e-3, beta=1e-2):
        us = us.squeeze(1)
        mr = mr.squeeze(1)

        bs = mr.size(0)
        pad = (mr.size(1) - (2*radius+1)) // 2
        count = (2*radius+1)**3

        self.f = self.f.to(mr.device)

        grad = torch.norm(F.conv3d(mr.unsqueeze(1), self.f, padding=1), dim=1)

        A = torch.ones(bs, 3, count, device=mr.device)
        A[:, 0] = mr[:, pad:-pad, pad:-pad, pad:-pad].reshape(bs, -1)
        A[:, 1] = grad[:, pad:-pad, pad:-pad, pad:-pad].reshape(bs, -1)
        b = us[:, pad:-pad, pad:-pad, pad:-pad].reshape(bs, -1)

        C = torch.einsum("bip,bjp->bij", A, A) / count + torch.eye(3, device=mr.device).unsqueeze(0) * alpha
        Atb = torch.einsum("bip,bp->bi", A, b) / count
        coeff = torch.linalg.solve(C, Atb)
        var = torch.mean(b**2, dim=1) - torch.mean(b, dim=1)**2
        dist = torch.mean(b**2, dim=1) + torch.einsum("bi,bj,bij->b", coeff, coeff, C) - 2*torch.einsum("bi,bi->b", coeff, Atb)
        sym = (var - dist)/var.clamp_min(beta)
        
        return sym.clamp(0, 1)
