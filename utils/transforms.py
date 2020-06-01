# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import cv2
import torch
import scipy
import scipy.misc
import numpy as np
import random


MATCHED_PARTS = {
    "300W": ([1, 17], [2, 16], [3, 15], [4, 14], [5, 13], [6, 12], [7, 11], [8, 10],
             [18, 27], [19, 26], [20, 25], [21, 24], [22, 23],
             [32, 36], [33, 35],
             [37, 46], [38, 45], [39, 44], [40, 43], [41, 48], [42, 47],
             [49, 55], [50, 54], [51, 53], [62, 64], [61, 65], [68, 66], [59, 57], [60, 56]),
    "AFLW": ([1, 6],  [2, 5], [3, 4],
             [7, 12], [8, 11], [9, 10],
             [13, 15],
             [16, 18]),
    "COFW": ([1, 2], [5, 7], [3, 4], [6, 8], [9, 10], [11, 12], [13, 15], [17, 18], [14, 16], [19, 20], [23, 24]),
    "WFLW": ([0, 32],  [1,  31], [2,  30], [3,  29], [4,  28], [5, 27], [6, 26], [7, 25], [8, 24], [9, 23], [10, 22],
             [11, 21], [12, 20], [13, 19], [14, 18], [15, 17],  # check
             [33, 46], [34, 45], [35, 44], [36, 43], [37, 42], [38, 50], [39, 49], [40, 48], [41, 47],  # elbrow
             [60, 72], [61, 71], [62, 70], [63, 69], [64, 68], [65, 75], [66, 74], [67, 73],
             [55, 59], [56, 58],
             [76, 82], [77, 81], [78, 80], [87, 83], [86, 84],
             [88, 92], [89, 91], [95, 93], [96, 97])}


def fliplr_joints(x, width, dataset='aflw'):
    """
    flip coords
    """
    matched_parts = MATCHED_PARTS[dataset]
    # Flip horizontal
    x[:, 0] = width - x[:, 0]

    if dataset == 'WFLW':
        for pair in matched_parts:
            tmp = x[pair[0], :].copy()
            x[pair[0], :] = x[pair[1], :]
            x[pair[1], :] = tmp
    else:
        for pair in matched_parts:
            tmp = x[pair[0] - 1, :].copy()
            x[pair[0] - 1, :] = x[pair[1] - 1, :]
            x[pair[1] - 1, :] = tmp
    return x


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def crop_v2(img, center, scale, output_size, translation=(1,1), rot=0):
    trans = get_affine_transform(center, scale, rot, output_size, translation)

    dst_img = cv2.warpAffine(
        img, trans, (int(output_size[0]), int(output_size[1])),
        flags=cv2.INTER_LINEAR
    )

    return dst_img


def get_transform(center, scale, output_size, translation=(1,1), rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(output_size[1]) / h
    t[1, 1] = float(output_size[0]) / h
    t[0, 2] = output_size[1] * (-float(center[0]) / h + .5) * translation[0]
    t[1, 2] = output_size[0] * (-float(center[1]) / h + .5) * translation[1]
    t[2, 2] = 1

    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -output_size[1]/2
        t_mat[1, 2] = -output_size[0]/2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t

def add_occlusion(img, max_size=102):
    h, w = img.shape[:-1]

    h_size = random.choice(range(1, max_size+1))
    w_size = random.choice(range(1, max_size+1))

    h_start = random.choice(range(0, h-h_size))
    w_start = random.choice(range(0, w-w_size))

    img_w_occlusion = img.copy()

    img_w_occlusion[h_start:h_start+h_size, w_start:w_start+w_size] = 0
    return img_w_occlusion


def transform_pixel(pt, center, scale, output_size, invert=0, translation=(1,1), rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, output_size, translation=translation, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(np.float32)


def transform_preds(coords, center, scale, output_size):

    for p in range(coords.size(0)):
        coords[p, 0:2] = torch.tensor(transform_pixel(coords[p, 0:2], center, scale, output_size, 1, (1,1), 0))
    return coords


def crop(img, center, scale, output_size, translation=(1,1), rot=0):
    center_new = center.clone()

    # Preprocessing for efficient cropping
    ht, wd = img.shape[0], img.shape[1]
    sf = scale * 200.0 / output_size[0]
    if sf < 2:
        sf = 1
    else:
        new_size = int(np.math.floor(max(ht, wd) / sf))
        new_ht = int(np.math.floor(ht / sf))
        new_wd = int(np.math.floor(wd / sf))
        if new_size < 2:
            return torch.zeros(output_size[0], output_size[1], img.shape[2]) \
                        if len(img.shape) > 2 else torch.zeros(output_size[0], output_size[1])
        else:
            img = scipy.misc.imresize(img, [new_ht, new_wd])  # (0-1)-->(0-255)
            center_new[0] = center_new[0] * 1.0 / sf
            center_new[1] = center_new[1] * 1.0 / sf
            scale = scale / sf

    # Upper left point
    ul = np.array(transform_pixel([0, 0], center_new, scale, output_size, invert=1, translation=translation)).astype(int)
    # Bottom right point
    br = np.array(transform_pixel(output_size, center_new, scale, output_size, invert=1, translation=translation)).astype(int)
    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]

    new_img = np.zeros(new_shape, dtype=np.float32)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]
    new_img = scipy.misc.imresize(new_img, output_size)
    return new_img


def generate_target(img, pt, sigma, label_type='Gaussian'):
    # Check that any part of the gaussian is in-bounds
    tmp_size = sigma * 3
    ul = [int(pt[0] - tmp_size), int(pt[1] - tmp_size)]
    br = [int(pt[0] + tmp_size + 1), int(pt[1] + tmp_size + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if label_type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    else:
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img



# Original Code : https://github.com/protossw512/AdaptiveWingLoss/blob/master/core/dataloader.py
from scipy import interpolate
import matplotlib.pyplot as plt

def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )

    # Get the RGB buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring (fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll (buf, 3, axis=2)
    return buf

class AddBoundary(object):
    def __init__(self, num_landmarks=68):
        self.num_landmarks = num_landmarks

    def __call__(self, target, tpts):
        landmarks_64 = np.array(tpts)
        if self.num_landmarks == 68:
            boundaries = {}
            boundaries['cheek'] = landmarks_64[0:17]
            boundaries['left_eyebrow'] = landmarks_64[17:22]
            boundaries['right_eyebrow'] = landmarks_64[22:27]
            boundaries['uper_left_eyelid'] = landmarks_64[36:40]
            boundaries['lower_left_eyelid'] = np.array([landmarks_64[i] for i in [36, 41, 40, 39]])
            boundaries['upper_right_eyelid'] = landmarks_64[42:46]
            boundaries['lower_right_eyelid'] = np.array([landmarks_64[i] for i in [42, 47, 46, 45]])
            boundaries['noise'] = landmarks_64[27:31]
            boundaries['noise_bot'] = landmarks_64[31:36]
            boundaries['upper_outer_lip'] = landmarks_64[48:55]
            boundaries['upper_inner_lip'] = np.array([landmarks_64[i] for i in [60, 61, 62, 63, 64]])
            boundaries['lower_outer_lip'] = np.array([landmarks_64[i] for i in [48, 59, 58, 57, 56, 55, 54]])
            boundaries['lower_inner_lip'] = np.array([landmarks_64[i] for i in [60, 67, 66, 65, 64]])
        elif self.num_landmarks == 98:
            boundaries = {}
            boundaries['cheek'] = landmarks_64[0:33]
            boundaries['left_eyebrow'] = landmarks_64[33:38]
            boundaries['right_eyebrow'] = landmarks_64[42:47]
            boundaries['uper_left_eyelid'] = landmarks_64[60:65]
            boundaries['lower_left_eyelid'] = np.array([landmarks_64[i] for i in [60, 67, 66, 65, 64]])
            boundaries['upper_right_eyelid'] = landmarks_64[68:73]
            boundaries['lower_right_eyelid'] = np.array([landmarks_64[i] for i in [68, 75, 74, 73, 72]])
            boundaries['noise'] = landmarks_64[51:55]
            boundaries['noise_bot'] = landmarks_64[55:60]
            boundaries['upper_outer_lip'] = landmarks_64[76:83]
            boundaries['upper_inner_lip'] = np.array([landmarks_64[i] for i in [88, 89, 90, 91, 92]])
            boundaries['lower_outer_lip'] = np.array([landmarks_64[i] for i in [76, 87, 86, 85, 84, 83, 82]])
            boundaries['lower_inner_lip'] = np.array([landmarks_64[i] for i in [88, 95, 94, 93, 92]])
        elif self.num_landmarks == 19:
            boundaries = {}
            boundaries['left_eyebrow'] = landmarks_64[0:3]
            boundaries['right_eyebrow'] = landmarks_64[3:5]
            boundaries['left_eye'] = landmarks_64[6:9]
            boundaries['right_eye'] = landmarks_64[9:12]
            boundaries['noise'] = landmarks_64[12:15]

        elif self.num_landmarks == 29:
            boundaries = {}
            boundaries['upper_left_eyebrow'] = np.stack([
                landmarks_64[0],
                landmarks_64[4],
                landmarks_64[2]
            ], axis=0)
            boundaries['lower_left_eyebrow'] = np.stack([
                landmarks_64[0],
                landmarks_64[5],
                landmarks_64[2]
            ], axis=0)
            boundaries['upper_right_eyebrow'] = np.stack([
                landmarks_64[1],
                landmarks_64[6],
                landmarks_64[3]
            ], axis=0)
            boundaries['lower_right_eyebrow'] = np.stack([
                landmarks_64[1],
                landmarks_64[7],
                landmarks_64[3]
            ], axis=0)
            boundaries['upper_left_eye'] = np.stack([
                landmarks_64[8],
                landmarks_64[12],
                landmarks_64[10]
            ], axis=0)
            boundaries['lower_left_eye'] = np.stack([
                landmarks_64[8],
                landmarks_64[13],
                landmarks_64[10]
            ], axis=0)
            boundaries['upper_right_eye'] = np.stack([
                landmarks_64[9],
                landmarks_64[14],
                landmarks_64[11]
            ], axis=0)
            boundaries['lower_right_eye'] = np.stack([
                landmarks_64[9],
                landmarks_64[15],
                landmarks_64[11]
            ], axis=0)
            boundaries['noise'] = np.stack([
                landmarks_64[18],
                landmarks_64[21],
                landmarks_64[19]
            ], axis=0)
            boundaries['outer_upper_lip'] = np.stack([
                landmarks_64[22],
                landmarks_64[24],
                landmarks_64[23]
            ], axis=0)
            boundaries['inner_upper_lip'] = np.stack([
                landmarks_64[22],
                landmarks_64[25],
                landmarks_64[23]
            ], axis=0)
            boundaries['outer_lower_lip'] = np.stack([
                landmarks_64[22],
                landmarks_64[26],
                landmarks_64[23]
            ], axis=0)
            boundaries['inner_lower_lip'] = np.stack([
                landmarks_64[22],
                landmarks_64[27],
                landmarks_64[23]
            ], axis=0)
        functions = {}

        for key, points in boundaries.items():
            temp = points[0]
            new_points = points[0:1, :]
            for point in points[1:]:
                if point[0] == temp[0] and point[1] == temp[1]:
                    continue
                else:
                    new_points = np.concatenate((new_points, np.expand_dims(point, 0)), axis=0)
                    temp = point
            points = new_points
            if points.shape[0] == 1:
                points = np.concatenate((points, points+0.001), axis=0)
            k = min(4, points.shape[0])
            functions[key] = interpolate.splprep([points[:, 0], points[:, 1]], k=k-1,s=0)

        boundary_map = np.zeros((64, 64))

        fig = plt.figure(figsize=[64/96.0, 64/96.0], dpi=96)

        ax = fig.add_axes([0, 0, 1, 1])

        ax.axis('off')

        ax.imshow(boundary_map, interpolation='nearest', cmap='gray')
        #ax.scatter(landmarks[:, 0], landmarks[:, 1], s=1, marker=',', c='w')

        for key in functions.keys():
            xnew = np.arange(0, 1, 0.01)
            out = interpolate.splev(xnew, functions[key][0], der=0)
            plt.plot(out[0], out[1], ',', linewidth=1, color='w')

        img = fig2data(fig)

        plt.close()

        sigma = 1
        temp = 255-img[:,:,1]
        temp = cv2.distanceTransform(temp, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        temp = temp.astype(np.float32)
        temp = np.where(temp < 3*sigma, np.exp(-(temp*temp)/(2*sigma*sigma)), 0 )

        fig = plt.figure(figsize=[64/96.0, 64/96.0], dpi=96)

        ax = fig.add_axes([0, 0, 1, 1])

        ax.axis('off')
        ax.imshow(temp, cmap='gray')
        plt.close()

        boundary_map = fig2data(fig)
        boundary_map = boundary_map[:,:,0][None,...] / 255.

        target = np.concatenate([target, boundary_map], axis=0)
        return target