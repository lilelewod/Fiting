import math
import numpy as np
# from estimators.ply.plyfile import PlyData
# from scipy.cluster._optimal_leaf_ordering import squareform
# import open3d as o3d

def save_point_cloud(cloud, path):
    cloud = np.asarray(cloud, dtype=np.float32)
    if cloud.shape[1] == 2:
        padding = np.zeros((cloud.shape[0], 1), dtype=cloud.dtype)
        cloud = np.hstack((cloud, padding))

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {cloud.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for point in cloud:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")


def save_triangle_mesh(vertices, faces, path, vertex_colors=None):
    try:
        import open3d as o3d
    except ModuleNotFoundError:
        print(f"skip saving triangle mesh to {path}: open3d is not installed")
        return

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)

    if vertices.shape[1] == 2:
        padding = np.zeros((vertices.shape[0], 1), dtype=vertices.dtype)
        vertices = np.hstack((vertices, padding))

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    if vertex_colors is not None:
        vertex_colors = np.asarray(vertex_colors, dtype=np.float64)
        if vertex_colors.max() > 1.0:
            vertex_colors = vertex_colors / 255.0
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(path, mesh)

def compute_resolution(cloud, rank=0.05, deduplicate=True):
    assert isinstance(cloud, np.ndarray)
    assert rank >= 0 and rank <= 1 # rank is a float in [0, 1]
    dimension = cloud.shape[1]    
    # import cupy
    # from cupyx.scipy.spatial import KDTree
    from sklearn.neighbors import KDTree        
    kdtree = KDTree(cloud)
    dists, ind = kdtree.query(cloud, k=2)
    dists = dists[:, 1]
    dists = np.sort(dists)
    resolution = dists[int(rank * dists.size)]  

    if deduplicate:
        import point_cloud_utils as pcu
        if dimension == 2:
            padding = np.zeros((cloud.shape[0],1))
            cloud = np.hstack((cloud, padding))
            p_dedup, idx_i, idx_j  = pcu.deduplicate_point_cloud(cloud, resolution)
            p_dedup = p_dedup[:, :2]
        elif dimension == 3:
            p_dedup, idx_i, idx_j  = pcu.deduplicate_point_cloud(cloud, resolution)
        else:
            assert False
    else:
        p_dedup = None   
    # if metric == 'median':
    #     resolution = np.median(dists)
    # elif metric == 'mininum':
    #     resolution = dists.min()
    # elif metric == 'average':
    #     resolution = dists.mean()
    # else:
    #     raise ValueError(f'Unknown metric: {metric}')
    # too_dense_points = (dists < 0.5* resolution).sum()
    # tmp = 0
    # median_resolution = cupy.asnumpy(cupy.median(dists[:, 1]))
    # print(f'resolution: {resolution}, median resolution: {median_resolution}')
    # average_resolution = cupy.asnumpy(dists[:, 1].mean())
    # print(f'resolution: {resolution}, average resolution: {average_resolution}')

    # # n_bins = 100
    # fig = plt.figure()
    # ax0 = fig.add_subplot(211)
    # ax0.hist(cupy.asnumpy(dists[:, 1]), bins=100)
    # ax1 = fig.add_subplot(212)
    # ax1.hist(cupy.asnumpy(dists[:, 1]), bins=50)

    # plt.show()    

    return resolution, p_dedup


# def show_image_as_cloud():
#     im = Image.open('test_1046.png')
#     im.show()
#     img = mpimg.imread('test_1046.png')
#     imgplot = plt.imshow(img)


# def show_image_as_cloud():
#     # im = Image.open('test_1046.png')
#     # im = Image.open('data/1.bmp')
#     # im = Image.open('data/test_003.png')
#     im = Image.open('data/000007.jpg').convert('L')
#     # im.show()
#     imarray = np.array(im)
#     if len(imarray.shape) > 2:
#         imarray = imarray[:, :, 0]
#
#     x = imarray.size
#     cloud = np.zeros((imarray.size, 3))
#     count = 0
#     for i in range(imarray.shape[0]):
#         for j in range(imarray.shape[1]):
#             cloud[count] = [i, j, imarray[i, j]/1.0]
#             count += 1
#
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(cloud)
#     # pcd.paint_uniform_color([0.5, 0.5, 0.5])
#     o3d.visualization.draw_geometries([pcd],
#                                       zoom=0.3412,
#                                       front=[0.4257, -0.2125, -0.8795],
#                                       lookat=[2.6172, 2.0475, 1.532],
#                                       up=[-0.0694, -0.9768, 0.2024])
#
#
#     # show_point_cloud(cloud)
#     tmp = 0
def cal_resolution(data):
    from scipy.spatial.distance import pdist, squareform

    dist = pdist(data)
    dist_matrix = squareform(dist)
    resolution = np.min(dist_matrix[np.nonzero(dist_matrix)])
    return resolution


def get_2D_rotation_matrix(angle, pivot):
    ca = math.cos(angle)
    sa = math.sin(angle)
    rotation = [[ca, -sa, (1-ca)*pivot[0]+sa*pivot[1]], [sa, ca, (1-ca)*pivot[1]-sa*pivot[0]]]
    return rotation


def show_image(img):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)
    plt.show()


def show_point_cloud(cloud, **kwargs):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if cloud.shape[1] == 3:
        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=1, c='g', marker='o')
    elif cloud.shape[1] == 2:
        z = np.zeros(cloud.shape[0])
        ax.scatter(cloud[:, 0], cloud[:, 1], z, s=1, c='g', marker='o')
    else:
        assert False
    set_axes_equal(ax)

    # view_init(90, -90):
    #   x-axis: left is small, right is big
    #   y-axis: bottom is small, top is big
    #   z-axis: far is small, near is big
    # (same as open3d)
    ax.view_init(90, -90)
    plt.show()

    # if cloud.shape[1] == 3:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.set_xlabel("x")
    #     ax.set_ylabel("y")
    #     ax.set_zlabel("z")
    #     ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=1, c='g', marker='o')
    #     set_axes_equal(ax)
    #
    #     # view_init(90, -90):
    #     #   x-axis: left is small, right is big
    #     #   y-axis: bottom is small, top is big
    #     #   z-axis: far is small, near is big
    #     # (same as open3d)
    #     ax.view_init(90, -90)
    #     plt.show()
    # elif cloud.shape[1] == 2:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.set_xlabel("x")
    #     ax.set_ylabel("y")
    #     ax.scatter(cloud[:, 0], cloud[:, 1], s=1, c='g', marker='o')
    #     # x_max = kwargs.get('x_max', 52)
    #     # y_max = kwargs.get('y_max', 52)
    #     # ax.set_xlim([0, x_max])
    #     # ax.set_ylim([0, y_max])
    #     plt.axis('equal')
    #     plt.show()
    # else:
    #     assert False
    # plt.show()


def gross_outlier(cloud, num=1, sd=0):
    assert num > 0
    (m, n) = cloud.shape
    num_outliers = int(m*num)
    outliers = np.zeros((num_outliers, n))
    for i in range(n):
        min_i = np.amin(cloud[:, i])
        max_i = np.amax(cloud[:, i])
        beyond = 0.5*(max_i-min_i)
        min_i = min_i-beyond
        max_i = max_i+beyond
        np.random.seed(sd+i)
        outliers[:, i] = min_i+(max_i-min_i)*np.random.rand(num_outliers)
    np.random.seed()
    return np.vstack((cloud, outliers))


def gaussian_noise(cloud, standard_deviation=1.0, seed=None):
    np.random.seed(seed=seed)
    noise = np.random.standard_normal(cloud.shape)*standard_deviation
    cloud += noise
    np.random.seed()
    return cloud


def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)


# def read_point_cloud(file_name):
#     file_type = file_name[-3:]
#     if 'xyz' == file_type:
#         x, y, z = [], [], []
#         with open(file_name, 'r') as f:
#             for line in f:
#                 point = line.split()
#                 x.append(float(point[0]))
#                 y.append(float(point[1]))
#                 z.append(float(point[2]))
#         x = np.array(x)
#         y = np.array(y)
#         z = np.array(z)
#         cloud = np.vstack((x, y, z)).transpose()
#     elif 'ply' == file_type:
#         ply_data = PlyData.read(file_name)
#         vertex = ply_data['vertex']
#         (x, y, z) = (vertex[t] for t in ('x', 'y', 'z'))
#         cloud = np.vstack((x, y, z)).transpose()
#     # elif 'ply' == file_type:
#     #     ply_data = PlyData.read(file_name)
#     #     vertex = ply_data['vertex']
#     #     (x, y, z, intensity) = (vertex[t] for t in ('x', 'y', 'z', 'intensity'))
#     #     cloud = np.vstack((x, y, z, intensity)).transpose()

#     return cloud


def cloud2state(cloud, boundary=None, voxel_size=0.5):
    if boundary is None:
        if cloud.shape[0] > 0:
            boundary = get_boundary(cloud)
        else:
            return -1
    voxel_numbers = np.zeros((3,), dtype='int')
    for i in range(3):
        voxel_numbers[i] = (boundary[1][i]-boundary[0][i])//voxel_size + 1
    state = np.zeros(voxel_numbers)
    state_index = np.zeros((3,), dtype=np.int32)
    inner_point_number = cloud.shape[0]
    for point in cloud:
        inner = True
        for i in range(3):
            if boundary[0][i] <= point[i] <= boundary[1][i]:
                state_index[i] = (point[i]-boundary[0][i])//voxel_size
            else:
                inner = False
                inner_point_number -= 1
                break
        if inner:
            state[tuple(state_index)] += 1
    state = state.reshape((state.size,))
    return state


def get_boundary(cloud):
    boundary = np.zeros((2, 3))
    n = cloud.shape[1]
    assert n == 3
    for i in range(n):
        boundary[0, i] = np.min(cloud[:, i])
        boundary[1, i] = np.max(cloud[:, i])
    return boundary

