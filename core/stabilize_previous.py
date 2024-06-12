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


def warp(x, flo, brightness=True):
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
    grid = torch.cat((xx, yy), 1).float()  # B,2 , H, W

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)  # B, H, W, 2
    if brightness:
        output = F.grid_sample(x, vgrid, mode='nearest', padding_mode='zeros').cpu().numpy()
    else:
        output = F.grid_sample(x, vgrid, padding_mode='zeros', mode='bilinear').cpu().numpy()
    return output


def load_image2(img, DEVICE):
    img = np.stack([img, img, img], axis=3)
    img = torch.from_numpy(img).permute(2, 3, 0, 1).float()
    return img.to(DEVICE)


def warp_flow2(flow, flow_to_warp):
    h, w = flow.shape[:2]
    flow_map_x, flow_map_y = np.meshgrid(np.arange(w), np.arange(h))
    flow = flow.cpu().numpy()
    # flow_to_warp = flow_to_warp.cpu().numpy()
    # Add flow_to_warp to the meshgrid to create the mapping
    map_x = (flow_map_x + flow_to_warp[:, :, 0]).astype(np.float32)
    map_y = (flow_map_y + flow_to_warp[:, :, 1]).astype(np.float32)

    # Warp the flow using the map
    warped_flow_x = cv2.remap(flow[:, :, 0], map_x, map_y, interpolation=cv2.INTER_LINEAR)
    warped_flow_y = cv2.remap(flow[:, :, 1], map_x, map_y, interpolation=cv2.INTER_LINEAR)

    warped_flow = np.dstack((warped_flow_x, warped_flow_y))
    return warped_flow



def accumulate_flows(flows):
    """
    Accumulate optical flows over a series of frames.

    :param flows: List of optical flows where flows[i] is the flow from frame i+1 to frame i.
    :return: The accumulated flow from the last frame to the first frame.
    """
    all_flows = []
    # Initialize with the first flow
    accumulated_flow = flows[0, :, :, :].cpu().numpy()
    all_flows.append(torch.tensor(accumulated_flow))

    for i in range(1, len(flows)):
        warped_flow = warp_flow2(flows[i], accumulated_flow)

        # Add the flows
        accumulated_flow[:, :, 0] += warped_flow[:, :, 0]
        accumulated_flow[:, :, 1] += warped_flow[:, :, 1]
        all_flows.append(torch.tensor(accumulated_flow))

    return all_flows


def demo(args):
    DEVICE = args.device
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    path = args.file
    path_dist = args.save_file
    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        img_name = path
        assert img_name.endswith('nrrd'), "only support nrrd file"
        sequence, header = nrrd.read(img_name)  # (256, 256, 12, 4)
        sequence = np.transpose(sequence, (0, 1, 2, 3))  # (256, 256, 12, 4)
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
        # traced_cell = torch.jit.script(model)
        # traced_cell.save('./raft_modelv3.2.zip')
        # flow_up = accumulate_flows_torch(flow_up_back)
        channel_stack = []
        for c in range(sequence.shape[-1]):
            img_c = sequence[:, :, 1:, c]
            brightness = ((np.sum(img_c == 0) / (w * h * t)) > 0.3)
            img_c = load_image2(img_c, DEVICE)
            image_warped = warp(img_c, flow_up, brightness)
            # img = image_warped.transpose(0, 2, 3, 1)
            channel_stack.append(image_warped[:, 0, :, :])
        output_sequence = np.float32(np.stack(channel_stack, axis=1))
        origin_frame = sequence[:, :, 0, :]
        origin_frame = np.expand_dims(origin_frame, axis=2).transpose(2, 3, 0, 1)
        output_sequence = np.concatenate([origin_frame, output_sequence], axis=0)
        out = np.transpose(output_sequence, (2, 3, 0, 1))  # (256, 256, 12, 4)
        nrrd.write(path_dist, data=out, header=header)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default='../models/raft-things.pth')
    parser.add_argument('--file', help="dataset for evaluation")
    parser.add_argument('--save_file', help="dataset for evaluation")
    parser.add_argument('--device', help="to run on cpu or gpu, ro run on gpu, choose cuda", default='cpu')
    parser.add_argument('--channel', help="reference channel", default='32S')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--occlusion', action='store_true', help='predict occlusion masks')

    args = parser.parse_args()
    demo(args)
