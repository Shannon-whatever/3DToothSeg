# %%
import numpy as np  
import trimesh

# %%
diff_count = np.load('tmp/diff_count.npy')
import matplotlib.pyplot as plt

unique, counts = np.unique(diff_count, return_counts=True)

plt.figure(figsize=(6,4))
plt.bar(unique, counts, color="orange", edgecolor="black")
plt.xticks(range(9))
plt.xlabel("diff_count value")
plt.ylabel("Count")
plt.title("Distribution of diff_count")
plt.show()

# %%
threshold = 4
def save_boundary_ply_trimesh(pointcloud, boundary_labels, filename="tmp/boundary_trimesh.ply"):
    coords = pointcloud[:, 6:9]
    labels = boundary_labels

    # RGB 0-255
    colors = np.zeros((coords.shape[0], 3), dtype=np.uint8)
    colors[labels == 1] = [255, 0, 0]   # red
    colors[labels == 0] = [0, 0, 255]   # blue

    cloud = trimesh.points.PointCloud(coords, colors=colors)
    cloud.export(filename)
    print(f"Saved to {filename}")


pointcloud = np.load('tmp/pointcloud.npy')
# boundary_labels = np.load('tmp/boundary_labels.npy')
boundary_labels = diff_count > threshold
save_boundary_ply_trimesh(pointcloud, boundary_labels)



# %%
