import os
import trimesh
import numpy as np
import json
from utils.other_utils import FDI2label, label2color_upper, label2color_lower
from utils.other_utils import face_labels_to_vertex_labels, output_pred_ply

from prepare_data.mesh_render import render


class DatasetPrepare():

    def __init__(self, root: str = '.datasets/teeth3ds', processed_folder: str = 'processed',
                 render_size=(1024, 1024), render_step=(6, 9)):
        
        self.root = root
        self.processed_folder = processed_folder
        self.mesh_view = ['upper', 'lower']
        self.render_size = render_size
        self.render_step = render_step


    def prepare_gt_ply(self):

        for view in self.mesh_view:
            root_mesh_folder = os.path.join(self.root, view)
            for root, dirs, files in os.walk(root_mesh_folder):
                for file in files:
                    if file.endswith(".obj"):
                        file_name = file.replace('.obj', '')
                        file_id = file_name.split('_')[0]
                        file_view = file_name.split('_')[1]
                        file_save_dir = os.path.join(self.root, self.processed_folder, file_view, file_id)
                        file_save_path = os.path.join(file_save_dir, f'{file_name}_gt.ply')
                        os.makedirs(file_save_dir, exist_ok=True)

                        mesh = trimesh.load(os.path.join(root, file))
                        with open(os.path.join(root, file).replace('.obj', '.json')) as f:
                            data = json.load(f)
                        labels = np.array(data["labels"])
                        labels = labels[mesh.faces]
                        labels = labels[:, 0]
                        labels = np.array([FDI2label[label] for label in labels])


                        mask = []
                        for label in labels:
                            if 'upper' in file:
                                color = label2color_upper[label][2]  # label 是单个 int
                            elif 'lower' in file:
                                color = label2color_lower[label][2]
                            mask.append(color)
                        mask = np.array(mask, dtype=np.uint8)  # shape: (N, 3)

                        # get vertex mask              # shape: (n_vertices, 3)

                        vertex_labels = face_labels_to_vertex_labels(mesh.faces, labels, len(mesh.vertices))
                        vertex_mask = []
                        for label in vertex_labels:
                            if file_view == 'upper':
                                color = label2color_upper[label][2]  # label 是单个 int
                            elif file_view == 'lower':
                                color = label2color_lower[label][2]
                            vertex_mask.append(color)
                        vertex_mask = np.array(vertex_mask, dtype=np.uint8)  # shape: (N, 3)

                        point_coords = mesh.vertices                # shape: (n_vertices, 3)
                        face_info = mesh.faces                      # shape: (n_faces, 3)
                        output_pred_ply(mask, None, file_save_path, point_coords, face_info, vertex_mask)

                        # prepare render data
                        render(file_save_path, file_save_dir, rend_size=self.render_size, render_step=self.render_step)

if __name__ == '__main__':

    dataset_prepare = DatasetPrepare(root='.datasets/teeth3ds/sample', processed_folder='processed')
    dataset_prepare.prepare_gt_ply()