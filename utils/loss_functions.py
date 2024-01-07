import torch
import torch.nn.functional as F
import numpy as np
# import torch.nn as nn
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


# class GlobalNCC:
#     def __init__(self, kernel_size = 21)

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
