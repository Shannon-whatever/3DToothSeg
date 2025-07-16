import torch
import numpy as np

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
        pc = pc / m * self.radius                          # 缩放到单位球内
        return pc


    def __call__(self, data):
        points,  labels, faces = data
        pc = points.clone()
        pc[:, 0:3] = self.pc_normalize(pc[:, 0:3])
        return (pc, labels, faces)




class PointcloudSample(object):
    def __init__(self, total=16000, sample=10000):
        self.total = total
        self.sample = sample

    def __call__(self, data):
        points, labels, faces = data
        sample = np.random.permutation(self.total)[:self.sample]
        return (points[sample], labels[sample], faces[sample])