

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

def plot_nondeep_model_comb(fig, lossLocalNCC_output_list, lossRot_output_list, lossTrans_output_list, initial_sampled_frame_est_np, sampled_frame_estimated_np, frame_gt_np):
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(234)
    ax3 = fig.add_subplot(235)
    ax4 = fig.add_subplot(236)

    ax1.plot(lossLocalNCC_output_list, marker='o', color='blue')
    ax1.plot(lossRot_output_list, color='darkred', marker='o')
    ax1.plot(lossTrans_output_list, color='lightsalmon', marker='o')
    ax1.legend(['localNCC', 'Rot', 'Trans'])
    ax1.set_title('Loss')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('iteration')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid()

    ax2.imshow(frame_gt_np[0, 0, :, :], cmap = "gray")
    ax2.set_title("US frame (target)")

    ax3.imshow(initial_sampled_frame_est_np[0, 0, :, :], cmap = "gray")
    ax3.set_title("Resampled image (initial)")

    ax4.imshow(sampled_frame_estimated_np[0, 0, :, :], cmap = "gray")
    ax4.set_title("Resampled image (latest)")

    plt.show(block=False)
    plt.pause(0.5)
    plt.clf()
    # plt.close()
def plot_nondeep_model_image_comb_with_ITK(fig, sampled_frame_estimated_np, frame_gt_np, sampled_frame_itk):
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    

    ax1.imshow(frame_gt_np[0, 0, :, :], cmap = "gray")
    ax1.set_title("US frame (target)")

    ax2.imshow(sampled_frame_itk[0, 0, :, :], cmap = "gray")
    ax2.set_title("Resampled image (ITK-approach)")

    ax3.imshow(sampled_frame_estimated_np[0, 0, :, :], cmap = "gray")
    ax3.set_title("Resampled image (GPU-accelerated)")

    plt.show(block=False)
    plt.pause(0.5)
    plt.clf()

def plot_nondeep_model_comb_with_ITK(fig, lossLocalNCC_output_list, lossRot_output_list, lossTrans_output_list, initial_sampled_frame_est_np, sampled_frame_estimated_np, frame_gt_np, sampled_frame_itk, ROI=None):
    # fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(221)
    ax1_ = fig.add_subplot(223)
    ax2 = fig.add_subplot(243)
    ax3 = fig.add_subplot(244)
    ax4 = fig.add_subplot(247)
    ax5 = fig.add_subplot(248)

    ax1.plot(lossLocalNCC_output_list, marker='o', color='blue')
    ax1.legend(['localNCC'])
    ax1.set_title('Loss (localNCC)')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('iteration')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid()
    # ax1_ = ax1.twinx()
    ax1_.plot(lossRot_output_list, color='darkred', marker='o')
    ax1_.plot(lossTrans_output_list, color='lightsalmon', marker='o')
    ax1_.legend(['Rotation(deg)', 'Translation(mm)'])
    ax1_.tick_params(axis='y', labelcolor='tab:red')
    ax1_.set_title('Loss(rot + trans)')
    ax1_.set_ylabel('loss')
    ax1_.set_xlabel('iteration')
    ax1_.grid()

    if ROI==None:
        ax2.imshow(frame_gt_np[0, 0, :, :], cmap = "gray")
        ax2.set_title("US frame (target)")

        ax3.imshow(initial_sampled_frame_est_np[0, 0, :, :], cmap = "gray")
        ax3.set_title("Resampled image (initial)")

        ax4.imshow(sampled_frame_itk[0, 0, :, :], cmap = "gray")
        ax4.set_title("Resampled image \n(registered by ITK)")

        ax5.imshow(sampled_frame_estimated_np[0, 0, :, :], cmap = "gray")
        ax5.set_title("Resampled image \n(gpu + local NCC (latest))")
    else:
        ax2.imshow(frame_gt_np[0, 0, :, :], cmap = "gray")
        ax2.add_patch(Rectangle((float(ROI[0]['W_min']), float(ROI[0]['H_min'])), float(ROI[0]['W_max'] - ROI[0]['W_min']+1), float(ROI[0]['H_max'] - ROI[0]['H_min']+1), edgecolor = 'red', facecolor='none'))

        ax2.set_title("US frame (target)")

        ax3.imshow(initial_sampled_frame_est_np[0, 0, :, :], cmap = "gray")
        ax3.set_title("Resampled image (initial)")

        ax4.imshow(sampled_frame_itk[0, 0, :, :], cmap = "gray")
        ax4.set_title("Resampled image \n(registered by ITK)")

        ax5.imshow(sampled_frame_estimated_np[0, 0, :, :], cmap = "gray")
        ax5.add_patch(Rectangle((float(ROI[0]['W_min']), float(ROI[0]['H_min'])), float(ROI[0]['W_max'] - ROI[0]['W_min']+1), float(ROI[0]['H_max'] - ROI[0]['H_min']+1), edgecolor = 'red', facecolor='none'))
        ax5.set_title("Resampled image \n(gpu + local NCC (latest))")

    plt.show(block=True)
    plt.pause(0.1)
    plt.clf()
    # plt.close()

def read_object_from_txt(fileName, object):
    # f = open(fileName, "r")
    object_array = []
    with open(fileName, 'r') as fp:
        for count, line in enumerate(fp):
            line_ = line.split("\n")
            x = line_[0].split(", ")
            output_dict = {}
            for i in range(len(x)):
                item = x[i].split(": ")
                output_dict[item[0]] = item[1]
            object_array.append(float(output_dict[object]))
        
    return object_array


