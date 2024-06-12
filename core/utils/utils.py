import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
from torch.types import Device
from typing import Optional
class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy

    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    # if mask:
    #     mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
    #     return img, mask.float()

    return img


def coords_grid(batch:int, ht:int, wd:int, device:torch.device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def warp_flow_torch(flow, flow_to_warp):
    h, w = flow.shape[1:]
    grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w))
    grid = torch.stack((grid_x, grid_y), 2).float().to(flow.device)  # Shape: (h, w, 2)

    # Add flow_to_warp to the meshgrid to create the mapping
    map_x = grid[:, :, 0] + flow_to_warp[0, :, :]
    map_y = grid[:, :, 1] + flow_to_warp[1, :, :]

    # Normalize the map to [-1, 1]
    map_x = 2.0 * map_x / (w - 1) - 1.0
    map_y = 2.0 * map_y / (h - 1) - 1.0
    map_xy = torch.stack((map_x, map_y), dim=2)  # Shape: (h, w, 2)
    map_xy = map_xy.unsqueeze(0)  # Add batch dimension, Shape: (1, h, w, 2)

    # Warp the flow using the map
    flow = flow.unsqueeze(0)  # Add batch dimension
    warped_flow_x = F.grid_sample(flow[:, :, :, :], map_xy, mode='bilinear', padding_mode='zeros')
    return warped_flow_x[0]  # Remove batch dimension


def accumulate_flows_torch(flows):
    """
    Accumulate optical flows over a series of frames.

    :param flows: List of optical flows where flows[i] is the flow from frame i+1 to frame i.
    :return: The accumulated flow from the last frame to the first frame.
    """
    all_flows = []
    # Initialize with the first flow
    accumulated_flow = flows[0, :, :, :].clone()

    all_flows.append(accumulated_flow.clone())
    i = 1
    while i < flows.shape[0]:
        warped_flow = warp_flow_torch(flows[i, :, :, :].clone(), accumulated_flow.clone())
        # Add the flows
        accumulated_flow += warped_flow
        all_flows.append(accumulated_flow.clone())
        i += 1
    return torch.stack(all_flows, dim=0)
