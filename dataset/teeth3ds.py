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
from utils.other_utils import rgb_mask_to_label




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
                    data_util.PointcloudRandomRotate(angle_sigma=0.06, angle_clip=0.18),
                    data_util.PointcloudRandomShift(shift_range=0.1),
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
                    l_view = l.split('_')[1]

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
            files_processed.extend(filter_files(process_mesh_folder, '_process.ply'))
            files_raw.extend(filter_files(raw_mesh_folder, 'obj'))
        return len(files_processed) == len(files_raw)

    def _process(self):
        # for f in filter_files(os.path.join(self.root, self.processed_folder), 'ply'):
        #     file_name = f.replace('.ply', '')
        #     file_id = file_name.split('_')[0]
        #     file_view = file_name.split('_')[1]
        #     if os.path.exists(os.path.join(self.root, self.processed_folder, file_view, file_id, f)):
        #         os.remove(os.path.join(self.root, self.processed_folder, file_view, file_id, f))

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
        
        file_name = f'{file_path.split("/")[-1]}_{file_path.split("/")[-2]}'
        mesh_path = os.path.join(file_path, f'{file_name}_process.ply')
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
        image_path_ls = sorted(glob(os.path.join(file_path, 'render', '*.png'))) 
        label_path_ls = sorted(glob(os.path.join(file_path, 'mask', '*.png')))
        renders, masks = [], []
        for image_path, label_path in zip(image_path_ls, label_path_ls):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
            image = np.float32(image)
            label = cv2.imread(label_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray with shape H * W * 3
            label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
            label = np.float32(label)
            if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
                raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
            if self.image_transform is not None:
                image, label = self.image_transform(image, label)


            # label color to label id
            label = rgb_mask_to_label(label)
            renders.append(image)
            masks.append(label)
        
        renders = torch.stack(renders, dim=0)  # (N_v, 3, H, W)
        masks = torch.stack(masks, dim=0)  # (N_v, H, W)

        # cameras
        with open(os.path.join(file_path, f'{file_name}_view.json'), 'r') as f:
            camera_params = json.load(f)

        cameras_rt = torch.stack([torch.tensor(cam["Rt"], dtype=torch.float32) for cam in camera_params])
        cameras_k = torch.stack([torch.tensor(cam["K"], dtype=torch.float32) for cam in camera_params])

        return_dict = {
            "pointcloud": pointcloud, # (N_pc, 9) face center coord norm + face normal + face center coord ori
            "labels": labels, # (N_pc)
            "point_coords": point_coords, # (N_vertices, 3) array
            "face_info": face_info, # (N_pc, 6) array 
            "renders": renders, # (N_v, 3, H, W)
            "masks": masks, # (N_v, H, W)
            "cameras_Rt": cameras_rt, # (N_v, 4, 4)
            "cameras_K": cameras_k # (N_v, 3, 3)
        }

        return return_dict
    
    def _load_in_memory(self):
        for f in tqdm(self.file_names, desc="Loading point clouds into memory"):
            data_dict = self._get_data(f)
            file_name = f'{f.split("/")[-1]}_{f.split("/")[-2]}'
            data_dict.update({"file_names": file_name})
            self.in_memory_data.append(data_dict)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        if self.in_memory:
            data_dict = self.in_memory_data[index]
        
        else:
            f = self.file_names[index]
            data_dict = self._get_data(f)
            file_name = f'{f.split("/")[-1]}_{f.split("/")[-2]}'
            data_dict.update({"file_names": file_name})

        return data_dict

            
        
        

if __name__ == "__main__":
# 
    train = Teeth3DSDataset(root = ".datasets/teeth3ds/sample", 
                            processed_folder='processed',
                            in_memory=True,
                            force_process=True, is_train=True, 
                            train_test_split=0)
