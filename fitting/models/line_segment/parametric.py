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
        self.x1 = None
        self.y1 = None


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
        # x0, y0, x1, y1
        self.lower_bound = np.asarray([-1.0, -1.0, -1.0, -1.0])
        self.upper_bound = np.asarray([1.0, 1.0, 1.0, 1.0])

        min_point = self.estimator.min_point
        assert 2 == min_point.size
        max_point = self.estimator.max_point
        bounding_box = max_point - min_point
        beyond = bounding_box / 10.0  # 10% of the bounding box, can be adjusted        
        self.lower_bound[0:2] = min_point[0:2] - beyond[0:2]
        self.upper_bound[0:2] = max_point[0:2] + beyond[0:2]
        self.lower_bound[2:4] = min_point[0:2] - beyond[0:2]
        self.upper_bound[2:4] = max_point[0:2] + beyond[0:2]                 
            
    @staticmethod
    def measure(trait):
        p0 = np.array([trait.x0, trait.y0])
        p1 = np.array([trait.x1, trait.y1])
        return np.linalg.norm(p1 - p0)

    def compute_top_dividing_level(self):
        length = self.measure(self.trait)
        level = math.log2(1 + length / self.estimator.resolution)
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
        trait.x1 = trait_flat[2]        
        trait.y1 = trait_flat[3]
        self.action = action
        self.trait = trait
        self.compute_top_dividing_level()        
        return trait

    def generate(self):
        self.compute_current_dividing_level()
        cloud = self.sample()
        self.estimator.add_model(new_measure=self.measure(self.trait), new_model=cloud)
        return cloud
    
    def sample(self):
        trait = self.trait
        level = self.current_dividing_level[0]
        t = np.arange(1/2**(level+1), 1, 1/2**level)
        x = trait.x0 + t * (trait.x1 - trait.x0)
        y = trait.y0 + t * (trait.y1 - trait.y0)
        cloud = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
        return cloud       
