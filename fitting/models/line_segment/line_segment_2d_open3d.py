from ..rule import ModelRule
import numpy as np
from tools.normalize import normalize
from copy import deepcopy
import math
import open3d as o3d
from easydict import EasyDict


class LineSegmentTrait(EasyDict):
    def __init__(self):
        EasyDict.__init__(self)        
        self.x0 = None
        self.y0 = None
        self.angle_z = None
        self.length = None


class LineSegmentRule(ModelRule):

    def __init__(self, estimator=None):
        ModelRule.__init__(self, estimator)
        self.lower_bound = None
        self.upper_bound = None          
        self.action = None
        self.trait = None
        self.set_trait_range()  

    def set_trait_range(self):
        # indices of 0, 1, 2, 3 corresponding to:
        # translation_x, translation_y, angle_z, length
        self.lower_bound = np.asarray([-1.0, -1.0, 0., 0.])
        self.upper_bound = np.asarray([1.0, 1.0, 2.0*np.pi, 1.])

        min_point = self.estimator.min_point
        max_point = self.estimator.max_point
        bounding_box = max_point - min_point
        max_length = np.linalg.norm(bounding_box)
        beyond = bounding_box / 10.0  # 10% of the bounding box, can be adjusted
        # assert 3 == min_point.size
        self.lower_bound[0:2] = min_point[0:2] - beyond[0:2]
        self.upper_bound[0:2] = max_point[0:2] + beyond[0:2]
        # the minimum length had better be larger than data resolution            
        self.lower_bound[-1] = 2 * self.estimator.data_resolution  
        self.upper_bound[-1] = max_length            
            
    @staticmethod
    def measure(trait):
        return trait.length

    def compute_top_dividing_level(self):
        level = math.log2(1 + self.trait.length / self.estimator.resolution)
        assert level >= 0
        self.top_level = np.asarray([math.floor(level)])

    def get_num_variables(self):
        assert self.lower_bound.size == self.upper_bound.size
        return self.lower_bound.size

    def parse(self, **kwargs):
        action = kwargs['action']
        assert action.size == self.get_num_variables()
        trait_flat = normalize(action, self.lower_bound, self.upper_bound)
        trait = LineSegmentTrait()
        trait.x0 = trait_flat[0]
        trait.y0 = trait_flat[1]
        trait.angle_z = trait_flat[2]        
        trait.length = trait_flat[3]
        self.action = action
        self.trait = trait
        self.compute_top_dividing_level()        
        return trait

    def generate(self):
        self.compute_current_dividing_level()
        open3d_version = self.estimator.cfg['estimator'].get('open3d_version', 'legacy')
        if 'legacy' == open3d_version:
            cloud = self.sample_legacy()
        else:
            cloud = self.sample_tensor()
        self.estimator.add_model(new_measure=self.measure(self.trait), new_model=cloud)
        return cloud
    
    def sample_legacy(self):
        trait = self.trait
        level = self.current_dividing_level[0]
        i = np.arange(1/2**(level+1), 1, 1/2**level)
        x = trait.length * i
        x = x[:,np.newaxis]
        yz = np.zeros((x.shape[0], 2))
        points = np.hstack((x, yz))

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)

        rotation = o3d.geometry.get_rotation_matrix_from_xyz((0., 0., trait.angle_z))
        cloud.rotate(rotation, center=(0, 0, 0))
        cloud.translate((trait.x0, trait.y0, 0.))
        return cloud

    def sample_tensor(self):
        trait = self.trait
        level = self.current_dividing_level[0]
        i =o3d.core.Tensor.arange(1/2**(level+1), 1, 1/2**level, dtype=o3d.core.Dtype.Float32, device=self.estimator.device)
        x = i * trait.length
        # x.shape.append(1)
        # x.reshape((x.shape[0], 1))
        # x = x[:,np.newaxis]
        # yz = np.zeros((x.shape[0], 2))
        # points = np.hstack((x, yz))

        points = o3d.core.Tensor.zeros((x.shape[0], 3), dtype=o3d.core.Dtype.Float32, device=self.estimator.device)
        points[:,0] = x

        # cloud = o3d.geometry.PointCloud()
        # cloud.points = o3d.utility.Vector3dVector(points)

        # cloud = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
        cloud = o3d.t.geometry.PointCloud(points)
        rotation = o3d.geometry.get_rotation_matrix_from_xyz((0., 0., trait.angle_z))
        cloud.rotate(rotation, center=(0, 0, 0))
        cloud.translate((trait.x0, trait.y0, 0.))
        return cloud.point.positions           

    @staticmethod
    def example_traits(num_models, data_resolution):
        traits = []
        trait_0 = LineSegmentTrait()
        trait_0.x0, trait_0.y0 = 0., 0.
        trait_0.length, trait_0.angle_z = 2., np.pi/2
        for i in range(num_models):
            trait = deepcopy(trait_0)
            trait.x0 += 3 * i * data_resolution
            traits.append(trait)
        return traits
