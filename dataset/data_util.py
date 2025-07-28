import torch
import numpy as np
from .pointaug_provider import rotate_perturbation_point_cloud_with_normal, shift_point_cloud


class PointcloudToTensor(object):
    def __call__(self, data):
        points, labels, faces = data
        return torch.from_numpy(points).float(), torch.from_numpy(labels).long(), faces


class PointcloudNormalize(object):
    def __init__(self, radius=1):
        self.radius = radius

    def pc_normalize(self, pc):
        centroid = pc.mean(dim=0)                 # (3,)
        pc = pc - centroid                        # 中心化
        m = torch.sqrt((pc ** 2).sum(dim=1)).max()  # 最大欧几里得距离
        pc = pc / m * self.radius                   # 缩放到单位球内
        return pc


    def __call__(self, data):
        points, labels, faces = data
        # pc = points.clone()
        # pc[:, 0:3] = self.pc_normalize(points[:, 0:3])  # 只对坐标进行归一化处理

        pc = self.pc_normalize(points[:, 0:3])
        points_new = torch.cat([pc, points[:, 3:6], points[:, 0:3]], dim=-1) # norm coords, normals, original coords

        return (points_new, labels, faces)


class PointcloudRandomRotate(object):
    """ 使用 rotate_perturbation_point_cloud_with_normal 实现带法向量点云的小角度随机旋转 """
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma = angle_sigma
        self.angle_clip = angle_clip

    def __call__(self, data):
        points, labels, face_info = data

        # points_np = points.numpy()
        if points.ndim == 2:
            points = points[None, ...]  # 增加batch维度

        rotated_points = rotate_perturbation_point_cloud_with_normal(
            points, self.angle_sigma, self.angle_clip)

        if rotated_points.shape[0] == 1:
            rotated_points = rotated_points[0]

        return (rotated_points, labels, face_info)

class PointcloudRandomShift(object):
    """ 使用 shift_point_cloud 实现随机平移 """
    def __init__(self, shift_range=0.1):
        self.shift_range = shift_range

    def __call__(self, data):
        points, labels, face_info = data
        # points_np = points.numpy()

        if points.ndim == 2:
            points = points[None, ...]

        shifted = shift_point_cloud(points, self.shift_range)

        if shifted.shape[0] == 1:
            shifted = shifted[0]

        return (shifted, labels, face_info)


class PointcloudSample(object):
    def __init__(self, total=16000, sample=10000, permute=True):
        self.total = total
        self.sample = sample
        self.permute = permute

    def __call__(self, data):
        points, labels, faces = data
        if self.permute:
            permute = np.random.permutation(self.total)[:self.sample]
            points = points[permute]
            labels = labels[permute]
            faces = faces[permute]
        else:
            if self.sample < self.total:
                points = points[:self.sample]
                labels = labels[:self.sample]
                faces = faces[:self.sample]

        return (points, labels, faces)
    

