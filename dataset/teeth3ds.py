from glob import glob
import json
import os
from pathlib import Path
import numpy as np
import pyfqmr
import trimesh
from scipy import spatial
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
import cv2
import torch


from dataset import data_util
from dataset import image_util
from utils.mesh_io import filter_files
from utils.other_utils import FDI2label, label2color_upper, label2color_lower, output_pred_ply, color2label, load_color_from_ply, face_labels_to_vertex_labels




class Teeth3DSDataset(Dataset):

    def __init__(self, root: str = '.datasets/teeth3ds', processed_folder: str = 'processed',
                 in_memory: bool = False,
                 force_process=False, train_test_split=1, is_train=True, 
                 num_points=16000, sample_points=16000):
        
        self.root = root
        self.processed_folder = processed_folder
        self.in_memory = in_memory
        self.in_memory_data = []
        self.file_names = []
        self.train_test_split = train_test_split
        self.mesh_view = ['upper', 'lower']

        self.num_points = num_points

        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]
        
        if is_train:
            self.point_transform = transforms.Compose(
                [
                    data_util.PointcloudToTensor(),
                    data_util.PointcloudNormalize(radius=1),
                    data_util.PointcloudSample(total=num_points, sample=sample_points, permute=True)
                ]
            )

            self.image_transform = image_util.Compose([
                image_util.RandScale([0.5, 2.0]), # args.scale_min, args.scale_max
                image_util.RandRotate([-10, 10], padding=mean, ignore_label=255),
                image_util.RandomGaussianBlur(),
                image_util.RandomHorizontalFlip(),
                image_util.Crop([465, 465], crop_type='rand', padding=mean, ignore_label=255),
                image_util.ToTensor(),
                image_util.Normalize(mean=mean, std=std)
            ])

        else:
            self.point_transform = transforms.Compose(
                [
                    data_util.PointcloudToTensor(),
                    data_util.PointcloudNormalize(radius=1),
                    data_util.PointcloudSample(total=num_points, sample=sample_points, permute=False)
                ]
            )

            self.image_transform = image_util.Compose([
                image_util.Crop([465, 465], crop_type='center', padding=mean, ignore_label=255),
                image_util.ToTensor(),
                image_util.Normalize(mean=mean, std=std)
            ])

        Path(os.path.join(self.root, self.processed_folder, 'upper')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.root, self.processed_folder, 'lower')).mkdir(parents=True, exist_ok=True)
        if not self._is_processed() or force_process:
            self._process()
        self._set_file_index(is_train)
        if self.in_memory:
            self._load_in_memory()


        
        
    def _set_file_index(self, is_train: bool):
        if self.train_test_split == 1:
            split_files = ['training_lower.txt', 'training_upper.txt'] if is_train else ['testing_lower.txt',
                                                                                         'testing_upper.txt']
        elif self.train_test_split == 2:
            split_files = ['public-training-set-1.txt', 'public-training-set-2.txt'] if is_train \
                else ['private-testing-set.txt']
        elif self.train_test_split == 0:
            split_files = ['training_lower_sample.txt', 'training_upper_sample.txt']
        else:
            raise ValueError(f'train_test_split should be 0, 1 or 2. not {self.train_test_split}')
        for f in split_files:
            with open(f'.datasets/teeth3ds/Teeth3DS_split/{f}') as file:
                for l in file:
                    l = f'{l.rstrip()}'
                    l_name = l.split('_')[0]
                    l_view = l_name.split('_')[1]

                    self.file_names.append(os.path.join(self.root, self.processed_folder, l_view, l_name))



    
    def _donwscale_mesh(self, mesh, labels):
        mesh_simplifier = pyfqmr.Simplify()
        mesh_simplifier.setMesh(mesh.vertices, mesh.faces)
        mesh_simplifier.simplify_mesh(target_count=16000, aggressiveness=3, preserve_border=True, verbose=0,
                                      max_iterations=2000)
        new_positions, new_face, _ = mesh_simplifier.getMesh()
        mesh_simple = trimesh.Trimesh(vertices=new_positions, faces=new_face)
        vertices = mesh_simple.vertices
        faces = mesh_simple.faces
        if faces.shape[0] < 16000:
            fs_diff = 16000 - faces.shape[0]
            faces = np.append(faces, np.zeros((fs_diff, 3), dtype="int"), 0)
        elif faces.shape[0] > 16000:
            mesh_simple = trimesh.Trimesh(vertices=vertices, faces=faces)
            samples, face_index = trimesh.sample.sample_surface_even(mesh_simple, 16000)
            mesh_simple = trimesh.Trimesh(vertices=mesh_simple.vertices, faces=mesh_simple.faces[face_index])
            faces = mesh_simple.faces
            vertices = mesh_simple.vertices
        mesh_simple = trimesh.Trimesh(vertices=vertices, faces=faces)

        mesh_v_mean = mesh.vertices[mesh.faces].mean(axis=1)
        mesh_simple_v = mesh_simple.vertices
        tree = spatial.KDTree(mesh_v_mean)
        query = mesh_simple_v[faces].mean(axis=1)
        distance, index = tree.query(query)
        labels = labels[index].flatten()
        return mesh_simple, labels

    def _count_total_obj_files(self):
        total = 0
        for view in self.mesh_view:
            root_mesh_folder = os.path.join(self.root, view)
            for _, _, files in os.walk(root_mesh_folder):
                total += sum(file.endswith(".obj") for file in files)
        return total

    def _iterate_mesh_and_labels(self):
    
        for view in self.mesh_view:
            root_mesh_folder = os.path.join(self.root, view)
            for root, dirs, files in os.walk(root_mesh_folder):
                for file in files:
                    if file.endswith(".obj"):
                        mesh = trimesh.load(os.path.join(root, file))
                        with open(os.path.join(root, file).replace('.obj', '.json')) as f:
                            data = json.load(f)
                        labels = np.array(data["labels"])
                        labels = labels[mesh.faces]
                        labels = labels[:, 0]
                        labels = np.array([FDI2label[label] for label in labels])
                        mesh, labels = self._donwscale_mesh(mesh, labels)
                        fn = file.replace('.obj', '')
                        yield mesh, labels, fn

    def _is_processed(self):
        files_raw, files_processed = [], []
        for view in self.mesh_view:
            raw_mesh_folder = os.path.join(self.root, view)
            process_mesh_folder = os.path.join(self.root, self.processed_folder, view)
            files_processed.extend(filter_files(process_mesh_folder, 'ply'))
            files_raw.extend(filter_files(raw_mesh_folder, 'obj'))
        return len(files_processed) == len(files_raw)

    def _process(self):
        for f in filter_files(os.path.join(self.root, self.processed_folder), 'ply'):
            if 'upper' in f:
                os.remove(os.path.join(self.root, self.processed_folder, 'upper', f))
            elif 'lower' in f:
                os.remove(os.path.join(self.root, self.processed_folder, 'lower', f))
        total_files = self._count_total_obj_files()

        with tqdm(total=total_files, desc="Processing meshes") as pbar:
            for view in self.mesh_view:
                root_mesh_folder = os.path.join(self.root, view)
                for root, dirs, files in os.walk(root_mesh_folder):
                    for file in files:
                        if file.endswith(".obj"):
                            file_name = file.replace('.obj', '')
                            file_id = file_name.split('_')[0]
                            file_view = file_name.split('_')[1]
                            mesh = trimesh.load(os.path.join(root, file))

                            with open(os.path.join(root, file).replace('.obj', '.json')) as f:
                                data = json.load(f)

                            labels = np.array(data["labels"])
                            labels = labels[mesh.faces]
                            labels = labels[:, 0]
                            labels = np.array([FDI2label[label] for label in labels])

                            mesh, labels = self._donwscale_mesh(mesh, labels)
                            fn = file.replace('.obj', '')

                            save_path = os.path.join(self.root, self.processed_folder, file_view, file_id,  f"{fn}_process.ply")

                            mask = []
                            for label in labels:
                                if 'upper' in fn:
                                    color = label2color_upper[label][2]  # label 是单个 int
                                elif 'lower' in fn:
                                    color = label2color_lower[label][2]
                                mask.append(color)
                            mask = np.array(mask, dtype=np.uint8)  # shape: (N, 3)

                            # get vertex mask              # shape: (n_vertices, 3)

                            vertex_labels = face_labels_to_vertex_labels(mesh.faces, labels, len(mesh.vertices))
                            vertex_mask = []
                            for label in vertex_labels:
                                if 'upper' in fn:
                                    color = label2color_upper[label][2]  # label 是单个 int
                                elif 'lower' in fn:
                                    color = label2color_lower[label][2]
                                vertex_mask.append(color)
                            vertex_mask = np.array(vertex_mask, dtype=np.uint8)  # shape: (N, 3)

                            point_coords = mesh.vertices                # shape: (n_vertices, 3)
                            face_info = mesh.faces                      # shape: (n_faces, 3)
                            output_pred_ply(mask, None, save_path, point_coords, face_info, vertex_mask)

                            pbar.update(1)


    def _get_data(self, file_path):
        
        mesh_path = f'{file_path}_process.ply'
        mesh = trimesh.load(mesh_path)

        point_coords = np.array(mesh.vertices)
        face_info = np.array(mesh.faces)


        cell_normals = np.array(mesh.face_normals)
        cell_coords = np.array([
            [
                (point_coords[face[0]][0] + point_coords[face[1]][0] + point_coords[face[2]][0]) / 3,
                (point_coords[face[0]][1] + point_coords[face[1]][1] + point_coords[face[2]][1]) / 3,
                (point_coords[face[0]][2] + point_coords[face[1]][2] + point_coords[face[2]][2]) / 3,
            ]
            for face in face_info
        ])

        pointcloud = np.concatenate((cell_coords, cell_normals), axis=1) # (N, 6)

        if pointcloud.shape[0] < self.num_points:
            padding = np.zeros((self.num_points - pointcloud.shape[0], pointcloud.shape[1]))
            face_info = np.concatenate((face_info, np.zeros(shape=(self.num_points - pointcloud.shape[0], 3))), axis=0)
            pointcloud = np.concatenate((pointcloud, padding), axis=0)

        # labels
        labels = load_color_from_ply(mesh_path)


        pointcloud, labels, face_info = self.point_transform([pointcloud, labels, face_info])

        # image
        image_path_ls = glob(os.path.join(file_path, 'render', '*.png')), 
        label_path_ls = glob(os.path.join(file_path, 'mask', '*.png'))
        renders, masks = [], []
        for image_path, label_path in zip(image_path_ls, label_path_ls):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
            image = np.float32(image)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
            if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
                raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
            if self.image_transform is not None:
                image, label = self.image_transform(image, label)
            renders.append(image)
            masks.append(label)
        
        renders = torch.stack(renders, dim=0)  # (N, C, H, W)
        masks = torch.stack(masks, dim=0)  # (N, H, W)


        return pointcloud, labels, point_coords, face_info, renders, masks
    
    def _load_in_memory(self):
        for f in tqdm(self.file_names, desc="Loading point clouds into memory"):
            pointcloud, labels, point_coords, faces, renders, masks = self._get_data(f)
            self.in_memory_data.append((pointcloud, labels, point_coords, faces, renders, masks, os.path.basename(f)))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        if self.in_memory:
            (pointcloud, labels, point_coords, faces, renders, masks, file_name) = self.in_memory_data[index]
        
        else:
            f = self.file_names[index]
            pointcloud, labels, point_coords, faces, renders, masks = self._get_data(f)
            file_name = os.path.basename(f)

        return pointcloud, labels, point_coords, faces, renders, masks, file_name

            
        
        

if __name__ == "__main__":
# 
    train = Teeth3DSDataset(root = ".datasets/teeth3ds/sample", 
                            processed_folder='processed',
                            in_memory=True,
                            force_process=True, is_train=True, 
                            train_test_split=0)
