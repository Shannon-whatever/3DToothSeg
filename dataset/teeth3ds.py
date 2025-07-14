import json
import os
import pickle
from os.path import join
from pathlib import Path

import numpy as np
import pyfqmr
import torch
import trimesh
from scipy import spatial
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.preprocessing_dilated import MoveToOriginTransform
from utils.mesh_io import filter_files
from utils.other_utils import FDI2label, label2color_upper, label2color_lower, output_pred_ply, color2label


def process_mesh(mesh: trimesh, labels: torch.tensor = None):
    mesh_faces = torch.from_numpy(mesh.faces.copy()).float()
    mesh_triangles = torch.from_numpy(mesh.vertices[mesh.faces]).float()
    mesh_face_normals = torch.from_numpy(mesh.face_normals.copy()).float()
    mesh_vertices_normals = torch.from_numpy(mesh.vertex_normals[mesh.faces]).float()
    if labels is None:
        labels = np.zeros(mesh_faces.shape[0], dtype=np.int64)
        for idx, face_color in enumerate(mesh.visual.face_colors):
            if len(face_color) == 4:
                face_color = face_color[:3]
            label = color2label[tuple(face_color)][-1]
            labels[idx] = label
    
        labels = torch.from_numpy(labels).long()
    return mesh_faces, mesh_triangles, mesh_vertices_normals, mesh_face_normals, labels


class Teeth3DSDataset(Dataset):

    def __init__(self, root: str, raw_folder: str = 'raw', processed_folder: str = 'processed_torch',
                 in_memory: bool = False, verbose: bool = True, pre_transform=None, post_transform=None,
                 force_process=False, train_test_split=1, is_train=True):
        self.root = root
        self.processed_folder = processed_folder
        self.raw_folder = raw_folder
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.in_memory = in_memory
        self.in_memory_data = []
        self.verbose = verbose
        self.file_names = []
        self.train_test_split = train_test_split
        self._set_file_index(is_train)
        self.move_to_origin = MoveToOriginTransform()
        Path(join(self.root, self.processed_folder)).mkdir(parents=True, exist_ok=True)
        Path(join(self.root, self.raw_folder)).mkdir(parents=True, exist_ok=True)
        if not self._is_processed() or force_process:
            self._process()
        self.processed_file_names = filter_files(join(self.root, self.processed_folder), 'pt')
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
            raise ValueError(f'train_test_split should be 1 or 2. not {self.train_test_split}')
        for f in split_files:
            with open(f'.datasets/teeth3ds/Teeth3DS_split/{f}') as file:
                for l in file:
                    l = f'{l.rstrip()}_process.ply'
                    if os.path.isfile(join(self.root, self.processed_folder, l)):
                        self.file_names.append(l)


    def _log(self, message: str):
        if self.verbose:
            print(message)

    def _loop(self, data):
        if self.verbose:
            return tqdm(data)
        return data
    
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

    def _iterate_mesh_and_labels(self):
        root_mesh_folder = join(self.root, self.raw_folder)
        for root, dirs, files in os.walk(root_mesh_folder):
            for file in files:
                if file.endswith(".obj"):
                    mesh = trimesh.load(join(root, file))
                    with open(join(root, file).replace('.obj', '.json')) as f:
                        data = json.load(f)
                    labels = np.array(data["labels"])
                    labels = labels[mesh.faces]
                    labels = labels[:, 0]
                    labels = [FDI2label[label] for label in labels]
                    mesh, labels = self._donwscale_mesh(mesh, labels)
                    fn = file.replace('.obj', '')
                    yield mesh, labels, fn

    def _is_processed(self):
        files_processed = filter_files(join(self.root, self.processed_folder), 'ply')
        files_raw = filter_files(join(self.root, self.raw_folder), 'obj')
        return len(files_processed) == len(files_raw)

    def _process(self):
        self._log('Processing data')
        # for f in filter_files(join(self.root, self.processed_folder), 'pt'):
        #     os.remove(join(self.root, self.processed_folder, f))
        for mesh, labels, fn in self._loop(self._iterate_mesh_and_labels()):
            save_path = f'{join(self.root, self.processed_folder)}/{fn}_process.ply'
            mask = []
            for label in labels:
                if 'upper' in fn:
                    color = label2color_upper[label][2]  # label 是单个 int
                elif 'lower' in fn:
                    color = label2color_lower[label][2]
                mask.append(color)
            mask = np.array(mask, dtype=np.uint8)  # shape: (N, 3)

            point_coords = mesh.vertices                # shape: (n_vertices, 3)
            face_info = mesh.faces                      # shape: (n_faces, 3)
            output_pred_ply(mask, None, save_path, point_coords, face_info)

            # mesh = self.move_to_origin(mesh)
            # data = process_mesh(mesh, torch.from_numpy(labels).long())
            # if self.pre_transform is not None:
            #     data = self.pre_transform(data)
            # with open(f'{join(self.root, self.processed_folder)}/data_{fn}.pt', 'wb') as f:
            #     pickle.dump(data, f)
        self._log('Processing done')

    # def _load_in_memory(self):
    #     files_processed = [join(self.root, self.processed_folder, f) for f in self.file_names]
    #     for i, f in enumerate(files_processed):
    #         file = open(join(self.root, self.processed_folder, f), 'rb')
    #         data = pickle.load(file)
    #         if self.post_transform is not None:
    #             data = self.post_transform(data)
    #         self.in_memory_data.append(data)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        if self.in_memory:
            raise NotImplementedError("In-memory loading is not implemented yet.")
            # return self.in_memory_data[index]
        else:
            f = self.file_names[index]
            file = open(join(self.root, self.processed_folder, f), 'rb')
            data = pickle.load(file)
            if self.post_transform is not None:
                data = self.post_transform(data)
            return data
        

# 
# train = Teeth3DSDataset("data/3dteethseg", processed_folder=f'processed',
#                                        verbose=True,
#                                        pre_transform=PreTransform(classes=17),
#                                        post_transform=None, in_memory=False,
#                                        force_process=False, is_train=True, train_test_split=train_test_split)
# test = Teeth3DSDataset("data/3dteethseg", processed_folder=f'processed',
#                                       verbose=True,
#                                       pre_transform=PreTransform(classes=17),
#                                       post_transform=None, in_memory=False,
#                                       force_process=False, is_train=False, train_test_split=train_test_split)