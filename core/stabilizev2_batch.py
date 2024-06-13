import sys

sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
import time

import torch.nn.functional as F
import os
import nrrd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def warp(x, flo, brightness=True, slice=0):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    if brightness:
        output = F.grid_sample(x, vgrid, mode='nearest', padding_mode='zeros').cpu().numpy()
    else:
        output = F.grid_sample(x, vgrid, padding_mode='zeros', mode='bilinear').cpu().numpy()
    return output

def load_image2(img, DEVICE):
    img = np.stack([img, img, img], axis=2)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def demo(args):
    DEVICE = args.device
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    path = args.path
    path_dist = args.save_path
    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        for img_name in os.listdir(path):
            if not img_name.endswith('nrrd'):
                continue
            try:
                sequence, header = nrrd.read(os.path.join(path, img_name))  # (256, 256, 12, 4)
            except:
                continue
            print('processing', img_name)
            sequence = np.transpose(sequence, (1, 0, 2, 3))  # (256, 256, 12, 4)
            output_sequence = [sequence[:, :, 0, :]]
            try:
                position_0 = header['Mims_mass_symbols'].split(' ').index(args.channel)
            except:
                position_0 = -1
            if position_0 != -1:
                wf_img = sequence[:, :, 1:, position_0]
                gt_img = sequence[:, :, :-1, position_0]
            else:
                wf_img = np.sum(sequence[:, :, 1:, :], axis=-1)
                gt_img = np.sum(sequence[:, :, :-1, :], axis=-1)
            h, w, t = wf_img.shape[:3]
            image1 = load_image2(gt_img, DEVICE)
            image2 = load_image2(wf_img, DEVICE)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            flow_up = model(image1, image2)
            channel_stack = []
            for c in range(sequence.shape[-1]):
                img_c = sequence[:, :, 1:, c]
                brightness = ((np.sum(img_c == 0) / (w * h * t)) > 0.3)
                img_c = load_image2(img_c, DEVICE)
                image_warped = warp(img_c, flow_up, brightness)
                channel_stack.append(image_warped[:, 0, :, :])
            output_sequence = np.float32(np.stack(channel_stack, axis=1))
            origin_frame = sequence[:, :, 0, :]
            origin_frame = np.expand_dims(origin_frame, axis=2).transpose(2, 3, 0, 1)
            output_sequence = np.concatenate([origin_frame, output_sequence], axis=0)
            out = np.transpose(output_sequence, (2, 3, 0, 1))  # (256, 256, 12, 4)
            nrrd.write(os.path.join(path_dist, img_name.replace('.nrrd', '_sta.nrrd')), data=out,
                       header=header)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default='../models/raft-things.pth')
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--save_path', help="dataset for evaluation")
    parser.add_argument('--device', help="to run on cpu or gpu", default='cpu')
    parser.add_argument('--channel', help="reference channel", default='32S')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--occlusion', action='store_true', help='predict occlusion masks')

    args = parser.parse_args()

    demo(args)
