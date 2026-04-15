def load_ply_data(data_path):
    assert data_path
    import open3d as o3d
    import numpy as np    
    print(f'input real-world data from {data_path}')
    # data = read_point_cloud(data_path)
    cloud = o3d.io.read_point_cloud(data_path)
    data = np.asarray(cloud.points)
    # data = self.compute_convexhull(data)
    return data


'''
Xiao G, Yu J, Ma J, Fan D-P, Shao L. Latent semantic consensus for deterministic geometric model fitting[J]. 
IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024, 46(9): 6139～6153.
'''
def load_lsc_data(estimator):
    cfg = estimator.cfg['estimator']    
    data_path = cfg['data_file']
    assert data_path
    import scipy.io as sio
    from .geometry import compute_resolution
  
    cloud = sio.loadmat(data_path, squeeze_me=True)
    print(f'input real-world data from {data_path}')
    data = cloud['data'].transpose()
    estimator.raw_data = data.copy()    

    estimator.data_resolution, data = compute_resolution(data.copy())
    estimator.min_point = data.min(0)            
    estimator.max_point = data.max(0)                       
    # model resolution should be smaller than 0.5 * data resolution
    estimator.model_resolution = cfg.get('model_resolution', 0.45 * estimator.data_resolution)
    assert estimator.model_resolution < 0.5 * estimator.data_resolution
    estimator.resolution = estimator.model_resolution
    return data


def load_lsc_data_as_3d(data_path):
    import numpy as np
    data = load_lsc_data(data_path)
    padding = np.zeros((data.shape[0],1))
    data = np.hstack((data, padding))
    return data


def load_image_data(estimator):
    import numpy as np    
    from PIL import Image        
    image_path = estimator.cfg['estimator']['data_file']
    print(f'input real-world data from {image_path}')
    img = Image.open(image_path)
    img = img.resize((int(img.size[0]/2), int(img.size[1]/2)))
    estimator.data_image = np.array(img)
    estimator.model_image = Image.new('L', estimator.data_image.shape)
    # self.model_image.save('blank.png')
    indexes = np.nonzero(estimator.data_image < 128)
    if indexes[0].size > 0:
        data = np.vstack((indexes[0], indexes[1])).transpose()
    else:
        assert False
    # show_point_cloud(data)
    estimator.raw_data = data.copy()
    estimator.data_resolution = 1.       
    return data
        
def load_3d_pointcloud_data(estimator):
    import open3d as o3d
    import numpy as np
    data_path = estimator.cfg['estimator']['data_file']
    
    # 使用 Open3D 读取点云
    pcd = o3d.io.read_point_cloud(data_path)
    if not pcd.has_points():
        raise ValueError(f"Failed to load point cloud or file is empty: {data_path}")
        
    points = np.asarray(pcd.points, dtype=np.float32)
    return points


def load_road_data(estimator):
    return load_3d_pointcloud_data(estimator)
