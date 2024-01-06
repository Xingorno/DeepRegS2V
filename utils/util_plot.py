

import matplotlib.pyplot as plt
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

def plot_nondeep_model_comb_with_ITK(fig, lossLocalNCC_output_list, lossRot_output_list, lossTrans_output_list, initial_sampled_frame_est_np, sampled_frame_estimated_np, frame_gt_np, sampled_frame_itk):
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

    ax2.imshow(frame_gt_np[0, 0, :, :], cmap = "gray")
    ax2.set_title("US frame (target)")

    ax3.imshow(initial_sampled_frame_est_np[0, 0, :, :], cmap = "gray")
    ax3.set_title("Resampled image (initial)")

    ax4.imshow(sampled_frame_itk[0, 0, :, :], cmap = "gray")
    ax4.set_title("Resampled image \n(registered by ITK)")

    ax5.imshow(sampled_frame_estimated_np[0, 0, :, :], cmap = "gray")
    ax5.set_title("Resampled image \n(gpu + local NCC (latest))")

    plt.show(block=False)
    # plt.pause(0.5)
    plt.clf()





