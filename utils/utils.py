import pdb
import time

import torch
import numpy as np
from thop import profile
from thop import clever_format
from skimage.measure import label, regionprops


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    if epoch == decay_epoch + 1:
        decay = decay_rate ** (epoch // decay_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses) - self.num, 0):]))


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))


@torch.no_grad()
def dequeue_and_enqueue(keys, queue, queue_ptr, queue_size):
    # gather keys before updating queue
    # pdb.set_trace()
    # keys = keys.detach().clone().cpu()
    # gathered_list = gather_together(keys)
    # keys = torch.cat(gathered_list, dim=0).cuda()  # 57*256

    batch_size = keys.shape[0]  # batch

    ptr = int(queue_ptr)

    queue[0] = torch.cat((queue[0], keys.detach()), dim=0)  # 把队列中剩下的和新生成的合并
    if queue[0].shape[0] >= queue_size:
        queue[0] = queue[0][-queue_size:, :]
        ptr = queue_size
    else:
        ptr = (ptr + batch_size) % queue_size  # move pointer

    queue_ptr[0] = ptr

    return batch_size


@torch.no_grad()
def dequeue_and_enqueue_prior(feats, keys, queue, queue_key, queue_time, queue_ptr, queue_size):
    ptr = int(queue_ptr)
    batch_size = feats.shape[0]  # batch 需要找出queue对应的key中最低的batchsize，替换成feats

    if queue[0].shape[0] != 0:
        timeout_mask = queue_time[0] <= 100
        # print('timeout_f:{}'.format(feats.shape))
        # print('timeout:{}'.format((~timeout_mask).sum()))
        queue[0] = queue[0][timeout_mask]
        queue_key[0] = queue_key[0][timeout_mask]
        queue_time[0] = queue_time[0][timeout_mask]
        ptr = ptr - (~timeout_mask).sum()
        # print('timeout_ptr:{}'.format(ptr))
        # print('timeout_qs:{}'.format(queue[0].shape))

    queue[0] = torch.cat((queue[0], feats.detach()), dim=0)  # 把队列中剩下的和新生成的合并
    queue_key[0] = torch.cat((queue_key[0], keys.detach()), dim=0)
    queue_time[0] = torch.cat((queue_time[0], torch.zeros_like(keys)), dim=0)

    sorted_keys, idx = torch.sort(queue_key[0], dim=0)  # 从小到大排序
    queue[0] = queue[0][idx]
    queue_key[0] = sorted_keys
    queue_time[0] = queue_time[0][idx]

    if queue[0].shape[0] >= queue_size:
        # print('out_q:{}'.format(queue[0].shape))
        queue[0] = queue[0][-queue_size:, :]
        queue_key[0] = queue_key[0][-queue_size:]
        queue_time[0] = queue_time[0][-queue_size:]
        ptr = queue_size
    else:
        ptr = (ptr + batch_size) % queue_size  # move pointer

    queue_ptr[0] = ptr
    queue_time[0] = queue_time[0] + 1
    return batch_size


# def init_point_sampling(mask, get_point=1):
#     """
#     Initialization samples points from the mask and assigns labels to them.
#     Args:
#         mask (torch.Tensor): Input mask tensor.
#         num_points (int): Number of points to sample. Default is 1.
#     Returns:
#         coords (torch.Tensor): Tensor containing the sampled points' coordinates (x, y).
#         labels (torch.Tensor): Tensor containing the corresponding labels (0 for background, 1 for foreground).
#     """
#     if isinstance(mask, torch.Tensor):
#         mask = mask.numpy()
#
#     # Get coordinates of black/white pixels
#     fg_coords = np.argwhere(mask == 1)[:, ::-1]
#     bg_coords = np.argwhere(mask == 0)[:, ::-1]
#
#     fg_size = len(fg_coords)
#     bg_size = len(bg_coords)
#
#     if get_point == 1:
#         if fg_size > 0:
#             index = np.random.randint(fg_size)
#             fg_coord = fg_coords[index]
#             label = 1
#         else:
#             index = np.random.randint(bg_size)
#             fg_coord = bg_coords[index]
#             label = 0
#         return torch.as_tensor([fg_coord.tolist()], dtype=torch.float), torch.as_tensor([label], dtype=torch.int)
#     else:
#         num_fg = get_point // 2
#         num_bg = get_point - num_fg
#         fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
#         bg_indices = np.random.choice(bg_size, size=num_bg, replace=True)
#         fg_coords = fg_coords[fg_indices]
#         bg_coords = bg_coords[bg_indices]
#         coords = np.concatenate([fg_coords, bg_coords], axis=0)
#         labels = np.concatenate([np.ones(num_fg), np.zeros(num_bg)]).astype(int)
#         indices = np.random.permutation(get_point)
#         coords, labels = torch.as_tensor(coords[indices], dtype=torch.float), torch.as_tensor(labels[indices],
#                                                                                               dtype=torch.int)
#         return coords, labels


# def get_boxes_from_mask(mask, box_num=1, std=0.1, max_pixel=5):
#     """
#     Args:
#         mask: Mask, can be a torch.Tensor or a numpy array of binary mask.
#         box_num: Number of bounding boxes, default is 1.
#         std: Standard deviation of the noise, default is 0.1.
#         max_pixel: Maximum noise pixel value, default is 5.
#     Returns:
#         noise_boxes: Bounding boxes after noise perturbation, returned as a torch.Tensor.
#     """
#     if isinstance(mask, torch.Tensor):
#         mask = mask.numpy()
#
#     label_img = label(mask)
#     regions = regionprops(label_img)
#
#     # Iterate through all regions and get the bounding box coordinates
#     boxes = [tuple(region.bbox) for region in regions]
#
#     # If the generated number of boxes is greater than the number of categories,
#     # sort them by region area and select the top n regions
#     if len(boxes) >= box_num:
#         sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)[:box_num]
#         boxes = [tuple(region.bbox) for region in sorted_regions]
#     elif len(boxes) == 0:
#         boxes = [(0, 0, *mask.shape)]
#     # If the generated number of boxes is less than the number of categories,
#     # duplicate the existing boxes
#     elif len(boxes) < box_num:
#         num_duplicates = box_num - len(boxes)
#         boxes += [boxes[i % len(boxes)] for i in range(num_duplicates)]
#
#     # Perturb each bounding box with noise
#     noise_boxes = []
#     for box in boxes:
#         y0, x0, y1, x1 = box
#         width, height = abs(x1 - x0), abs(y1 - y0)
#         # Calculate the standard deviation and maximum noise value
#         noise_std = min(width, height) * std
#         max_noise = min(max_pixel, int(noise_std * 5))
#         if max_noise == 0:
#             max_noise = 1
#         # Add random noise to each coordinate
#         noise_x = np.random.randint(-max_noise, max_noise)
#         noise_y = np.random.randint(-max_noise, max_noise)
#         x0, y0 = x0 + noise_x, y0 + noise_y
#         x1, y1 = x1 + noise_x, y1 + noise_y
#         noise_boxes.append((x0, y0, x1, y1))
#     return torch.as_tensor(noise_boxes, dtype=torch.float)

def init_point_sampling(mask, get_point=1):
    """
    Initialization samples points from the mask and assigns labels to them.
    Args:
        mask (torch.Tensor): Input mask tensor.
        num_points (int): Number of points to sample. Default is 1.
    Returns:
        coords (torch.Tensor): Tensor containing the sampled points' coordinates (x, y).
        labels (torch.Tensor): Tensor containing the corresponding labels (0 for background, 1 for foreground).
    """

    # Get coordinates of black/white pixels
    fg_coords = torch.nonzero(mask == 1)  # [:, ::-1]
    bg_coords = torch.nonzero(mask == 0)  # [:, ::-1]

    fg_size = len(fg_coords)
    bg_size = len(bg_coords)

    if get_point == 1:
        if fg_size > 0:
            index = torch.randint(fg_size, size=(1,))
            fg_coord = fg_coords[index]
            label = 1
        else:
            index = torch.randint(bg_size, size=(1,))
            fg_coord = bg_coords[index]
            label = 0
        return fg_coord, torch.as_tensor([label], dtype=torch.int)
    else:

        num_fg = get_point // 2
        if num_fg > fg_size:
            num_fg = fg_size
        num_bg = get_point - num_fg
        if num_bg > bg_size:
            num_bg = bg_size
            num_fg = get_point - num_bg
        if num_fg != 0:
            fg_indices = torch.randint(fg_size, size=(num_fg,))
            fg_coords = fg_coords[fg_indices]
        else:
            fg_coords = torch.empty(size=(1, 2)).cuda()
        if num_bg != 0:
            bg_indices = torch.randint(bg_size, size=(num_bg,))
            bg_coords = bg_coords[bg_indices]
        else:
            bg_coords = torch.empty(size=(1, 2)).cuda()
        coords = torch.cat([fg_coords, bg_coords], dim=0)
        labels = torch.cat([torch.ones(num_fg), torch.zeros(num_bg)], dim=0)
        indices = torch.randperm(get_point)
        return coords[indices], labels[indices]


# def init_point_sampling(mask, get_point=1):
#     """
#     备份
#     Initialization samples points from the mask and assigns labels to them.
#     Args:
#         mask (torch.Tensor): Input mask tensor.
#         num_points (int): Number of points to sample. Default is 1.
#     Returns:
#         coords (torch.Tensor): Tensor containing the sampled points' coordinates (x, y).
#         labels (torch.Tensor): Tensor containing the corresponding labels (0 for background, 1 for foreground).
#     """
#     if isinstance(mask, torch.Tensor):
#         mask = mask.numpy()
#
#     # Get coordinates of black/white pixels
#     fg_coords = np.argwhere(mask == 1)[:, ::-1]
#     bg_coords = np.argwhere(mask == 0)[:, ::-1]
#
#     fg_size = len(fg_coords)
#     bg_size = len(bg_coords)
#
#     if get_point == 1:
#         if fg_size > 0:
#             index = np.random.randint(fg_size)
#             fg_coord = fg_coords[index]
#             label = 1
#         else:
#             index = np.random.randint(bg_size)
#             fg_coord = bg_coords[index]
#             label = 0
#         return torch.as_tensor([fg_coord.tolist()], dtype=torch.float), torch.as_tensor([label], dtype=torch.int)
#     else:
#         num_fg = get_point // 2
#         num_bg = get_point - num_fg
#         fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
#         bg_indices = np.random.choice(bg_size, size=num_bg, replace=True)
#         fg_coords = fg_coords[fg_indices]
#         bg_coords = bg_coords[bg_indices]
#         coords = np.concatenate([fg_coords, bg_coords], axis=0)
#         labels = np.concatenate([np.ones(num_fg), np.zeros(num_bg)]).astype(int)
#         indices = np.random.permutation(get_point)
#         coords, labels = torch.as_tensor(coords[indices], dtype=torch.float), torch.as_tensor(labels[indices],
#                                                                                               dtype=torch.int)
#         return coords, labels


def get_boxes_from_mask(mask, box_num=1, std=0.1):
    """
    Args:
        mask: Mask, can be a torch.Tensor or a numpy array of binary mask. H*W
        box_num: Number of bounding boxes, default is 1.
        std: Standard deviation of the noise, default is 0.1.
        max_pixel: Maximum noise pixel value, default is 5.
    Returns:
        noise_boxes: Bounding boxes after noise perturbation, returned as a torch.Tensor.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    label_img = label(mask)
    regions = regionprops(label_img)

    # Iterate through all regions and get the bounding box coordinates
    boxes = [tuple(region.bbox) for region in regions]

    # If the generated number of boxes is greater than the number of categories,
    # sort them by region area and select the top n regions
    if len(boxes) >= box_num:
        sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)[:box_num]
        boxes = [tuple(region.bbox) for region in sorted_regions]
    if len(boxes) == 0:
        boxes = [(0, 0, *mask.shape)]
    if len(boxes) > 1:
        boxes = [tuple(region.bbox) for region in regions if region.area > 4]

    # Perturb each bounding box with noise
    noise_boxes = []
    for box in boxes:
        y0, x0, y1, x1 = box
        # width, height = abs(x1 - x0), abs(y1 - y0)
        # noise = min(width, height) * std
        # x0, y0 = max(x0 - noise, 0), max(y0 - noise, 0)
        # x1, y1 = min(x1 + noise, 383), min(y1 + noise, 383)
        noise_boxes.append((x0, y0, x1, y1))
    # pdb.set_trace()
    return torch.as_tensor(noise_boxes, dtype=torch.float)

# def get_boxes_from_mask_v2(mask1, mask2, box_num=10):
#     """
#     Args:
#         mask: Mask, can be a torch.Tensor or a numpy array of binary mask.
#         box_num: Number of bounding boxes, default is 1.
#         std: Standard deviation of the noise, default is 0.1.
#         max_pixel: Maximum noise pixel value, default is 5.
#     Returns:
#         noise_boxes: Bounding boxes after noise perturbation, returned as a torch.Tensor.
#     """
#     if isinstance(mask1, torch.Tensor):
#         mask1 = mask1.numpy()
#     if isinstance(mask2, torch.Tensor):
#         mask2 = mask2.numpy()
#
#     label_img1 = label(mask1)
#     regions1 = regionprops(label_img1)
#
#     label_img2 = label(mask2)
#     regions2 = regionprops(label_img2)
#
#     # Iterate through all regions and get the bounding box coordinates
#     boxes1 = [tuple(region.bbox) for region in regions1]
#     boxes2 = [tuple(region.bbox) for region in regions2]
#
#
#     # TODO 两个box要做配对
#
#
#
#
#     # If the generated number of boxes is greater than the number of categories,
#     # sort them by region area and select the top n regions
#     if len(boxes) >= box_num:
#         sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)[:box_num]
#         boxes = [tuple(region.bbox) for region in sorted_regions]
#     if len(boxes) == 0:
#         boxes = [(0, 0, *mask.shape)]
#
#     # Perturb each bounding box with noise
#     noise_boxes = []
#     for box in boxes:
#         y0, x0, y1, x1 = box
#         noise_boxes.append((x0, y0, x1, y1))
#     return torch.as_tensor(noise_boxes, dtype=torch.float)
