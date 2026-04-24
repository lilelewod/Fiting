def read_point_cloud(file_name):
    import numpy as np
    file_type = file_name[-3:]
    if 'xyz' == file_type:
        x, y, z = [], [], []
        with open(file_name, 'r') as f:
            for line in f:
                point = line.split()
                x.append(float(point[0]))
                y.append(float(point[1]))
                z.append(float(point[2]))
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        cloud = np.vstack((x, y, z)).transpose()
    elif 'ply' == file_type:
        try:
            from plyfile import PlyData

            ply_data = PlyData.read(file_name)
            vertex = ply_data['vertex']
            (x, y, z) = (vertex[t] for t in ('x', 'y', 'z'))
            cloud = np.vstack((x, y, z)).transpose()
        except ModuleNotFoundError:
            import open3d as o3d

            point_cloud = o3d.io.read_point_cloud(file_name)
            cloud = np.asarray(point_cloud.points)
    # elif 'ply' == file_type:
    #     ply_data = PlyData.read(file_name)
    #     vertex = ply_data['vertex']
    #     (x, y, z, intensity) = (vertex[t] for t in ('x', 'y', 'z', 'intensity'))
    #     cloud = np.vstack((x, y, z, intensity)).transpose()

    return cloud

def load_ply_data(estimator):
    cfg = estimator.cfg['estimator']    
    data_path = cfg['data_file']
    assert data_path    

    print(f'input real-world data from {data_path}')

    # import numpy as np
    # import open3d as o3d
    # cloud = o3d.io.read_point_cloud(data_path)
    # data = np.asarray(cloud.points)

    data = read_point_cloud(data_path)
    estimator.raw_data = data.copy()
    voxel_size_for_down_sampling = cfg.get('voxel_size_for_down_sampling', None)
    if voxel_size_for_down_sampling is None:
        assert 'data_resolution' in cfg
        assert 'model_resolution' in cfg
        estimator.data_resolution = cfg['data_resolution']
        estimator.model_resolution = cfg['model_resolution']        
    else:
        import point_cloud_utils as pcu
        data = pcu.downsample_point_cloud_on_voxel_grid(voxel_size_for_down_sampling, data)
        estimator.data_resolution = voxel_size_for_down_sampling
        estimator.model_resolution = 0.45 * estimator.data_resolution  # should be smaller than 0.5 * data_resolution        
    estimator.min_point = data.min(0)            
    estimator.max_point = data.max(0)    
    estimator.resolution = estimator.model_resolution
    return data

# def load_ply_data(data_path):
#     assert data_path
#     import open3d as o3d
#     import numpy as np    
#     print(f'input real-world data from {data_path}')
#     # data = read_point_cloud(data_path)
#     cloud = o3d.io.read_point_cloud(data_path)
#     data = np.asarray(cloud.points)
#     # data = self.compute_convexhull(data)
#     return data


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
        
