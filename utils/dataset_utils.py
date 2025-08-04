import torch


def custom_collate_fn(batch):
        pointclouds = []
        labels = []
        point_coords_list = []
        face_infos = []
        renders = []
        masks = []
        cameras_Rt = []
        cameras_K = []
        file_names = []

        for data_dict in batch:
            # 解包数据字典
            pc = data_dict['pointcloud']
            label = data_dict['labels']
            p_coords = data_dict['point_coords']
            f_info = data_dict['face_info']
            render = data_dict['renders']
            mask = data_dict['masks']
            camera_Rt = data_dict['cameras_Rt']
            camera_K = data_dict['cameras_K']
            file_name = data_dict['file_names']

            pointclouds.append(pc)
            labels.append(label)
            point_coords_list.append(p_coords)  # 不堆叠，保留为 list of np.array
            face_infos.append(f_info) # 保留为 list of np.array
            renders.append(render)
            masks.append(mask)
            cameras_Rt.append(camera_Rt)
            cameras_K.append(camera_K)
            file_names.append(file_name) # 保留为 list of str

        # 堆叠固定 shape 的数据
        pointclouds = torch.stack(pointclouds)  # (B, num_points, 6)
        labels = torch.stack(labels)            # (B, num_points)
        renders = torch.stack(renders)          # (B, num_views, 3, H, W)
        masks = torch.stack(masks)              # (B, num_views, H, W)
        cameras_Rt = torch.stack(cameras_Rt)    # (B, num_views, 4, 4)
        cameras_K = torch.stack(cameras_K)      # (B, num_views, 3, 3)

        return_dict = {
            'pointclouds': pointclouds,
            'labels': labels,
            'point_coords': point_coords_list,  # 保持为 list of np.array
            'face_infos': face_infos,
            'renders': renders,
            'masks': masks,
            'cameras_Rt': cameras_Rt,
            'cameras_K': cameras_K,
            'file_names': file_names
        }
        return return_dict