from ..rule import ModelRule, Token
import math
import numpy as np
import open3d as o3d
from tools.tool import rescale
import point_cloud_utils as pcu
from tools.geometry import show_point_cloud
from easydict import EasyDict


class RectangleTrait(EasyDict):  # trait denotes parameters
    def __init__(self):
        EasyDict.__init__(self)
        self.translation_x = 0.0
        self.translation_y = 0.0
        self.translation_z = 0.0
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0
        self.length_x = 0.5
        self.length_y = 2.0

    def toarray(self):
        return np.asarray(
            [
                self.translation_x,
                self.translation_y,
                self.translation_z,
                self.rotation_x,
                self.rotation_y,
                self.rotation_z,
                self.length_x,
                self.length_y,
            ]
        )


def array2trait(array):
    assert array.size == 8
    trait = RectangleTrait()
    trait.translation_x = array[0]
    trait.translation_y = array[1]
    trait.translation_z = array[2]
    trait.rotation_x = array[3]
    trait.rotation_y = array[4]
    trait.rotation_z = array[5]
    trait.length_x = array[6]
    trait.length_y = array[7]
    return trait


class RectangleRule(ModelRule):

    def __init__(self, estimator=None):
        ModelRule.__init__(self, estimator)
        self.lower_bound = None
        self.upper_bound = None
        self.lb = None
        self.ub = None
        self.trait = None
        self.action = None
        self.set_trait_range()

    def set_trait_range(self):
        self.lb = np.asarray([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.ub = np.asarray([1.0, 1.0, 1.0, 2 * np.pi, 2 * np.pi, 2 * np.pi, 1000.0, 1000.0])
        min_point = self.estimator.min_point
        max_point = self.estimator.max_point
        if min_point.size == 3:
            self.lb[0:3] = min_point - 1000.0
            self.ub[0:3] = max_point + 1000.0
        else:
            raise AssertionError("RectangleRule expects 3D point-cloud data.")
        self.lower_bound = array2trait(self.lb)
        self.upper_bound = array2trait(self.ub)

    @staticmethod
    def measure(trait):
        return trait.length_x * trait.length_y

    def compute_top_dividing_level(self):
        level = np.zeros(2)
        level[0] = math.log2(1 + self.trait.length_x / self.estimator.resolution)
        level[1] = math.log2(1 + self.trait.length_y / self.estimator.resolution)
        assert np.all(level >= 0)
        self.top_level = np.ceil(level)

    def generate(self):
        cloud = self.sample()
        token = Token(self.estimator.dimension)
        token.points = cloud
        token.trait = self.trait
        token.measure = self.measure(self.trait)
        token.action = self.action
        self.estimator.add_token(token)
        return cloud

    def sample(self):
        trait = self.trait
        level = self.compute_current_dividing_level()
        x = np.arange(1 / 2 ** (level[0] + 1), 1, 1 / 2 ** level[0])
        y = np.arange(1 / 2 ** (level[1] + 1), 1, 1 / 2 ** level[1])
        x, y = trait.length_x * x, trait.length_y * y
        x_size, y_size = x.size, y.size
        x = np.tile(x, y_size)
        y = np.repeat(y, x_size)
        z = np.zeros(x_size * y_size)
        cloud = np.vstack((x, y, z)).transpose()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        rotation = pcd.get_rotation_matrix_from_xyz(
            (trait.rotation_x, trait.rotation_y, trait.rotation_z)
        )
        pcd.rotate(rotation, center=(0, 0, 0))
        pcd.translate((trait.translation_x, trait.translation_y, trait.translation_z))
        return np.asarray(pcd.points)

    def sample_poisson_disk(self):
        trait = self.trait
        np_vertices = np.array(
            [
                [0, 0, 0],
                [0, trait.length_y, 0],
                [trait.length_x, trait.length_y, 0],
                [trait.length_x, 0, 0],
            ]
        ).astype(np.float32)
        np_triangles = np.array([[0, 1, 2], [0, 3, 2]]).astype(np.int32)
        f_i, bc = pcu.sample_mesh_poisson_disk(
            np_vertices, np_triangles, -1, self.estimator.resolution
        )
        v_poisson = pcu.interpolate_barycentric_coords(np_triangles, f_i, bc, np_vertices)
        show_point_cloud(v_poisson)
        return bc

    @staticmethod
    def example_traits():
        traits = []
        for model_index in range(3):
            trait = RectangleTrait()
            trait.translation_x = 0.8 * model_index
            trait.translation_z = 0.0
            traits.append(trait)
        return traits

    def get_num_variables(self):
        assert self.lb.size == self.ub.size
        return self.lb.size

    def parse(self, **kwargs):
        action = kwargs["action"]
        assert action.size == self.get_num_variables()
        trait_flat = rescale(action, self.lb, self.ub)
        self.trait = array2trait(trait_flat)
        self.action = action
        self.compute_top_dividing_level()
