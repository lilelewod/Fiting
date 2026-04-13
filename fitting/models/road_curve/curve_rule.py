from ..rule import ModelRule
import math
import numpy as np
from tools.tool import rescale
from easydict import EasyDict
from numba import float64, jit
from scipy.integrate import quad


def line_xy(trait, s):
    x = trait.x0 + s * np.cos(trait.azimuth0)
    y = trait.y0 + s * np.sin(trait.azimuth0)
    return x, y


def circle_xy(trait, s):
    x = trait.x0 + (np.sin(trait.azimuth0 + s * trait.horizontal_curvature0) - np.sin(trait.azimuth0)) / trait.horizontal_curvature0
    y = trait.y0 - (np.cos(trait.azimuth0 + s * trait.horizontal_curvature0) - np.cos(trait.azimuth0)) / trait.horizontal_curvature0
    return x, y


@jit(float64(float64, float64, float64, float64), nopython=True, cache=True)
def spiral_integrand_x(x, azimuth0, horizontal_curvature0, horizontal_curvature_gradient):
    return np.cos(azimuth0 + horizontal_curvature0 * x + horizontal_curvature_gradient * (x * x) / 2.)


@jit(float64(float64, float64, float64, float64), nopython=True, cache=True)
def spiral_integrand_y(x, azimuth0, horizontal_curvature0, horizontal_curvature_gradient):
    return np.sin(azimuth0 + horizontal_curvature0 * x + horizontal_curvature_gradient * (x * x) / 2.)


# @jit
# def spiral_xy(s, azimuth0, horizontal_curvature0, horizontal_curvature_gradient, x0, y0):
def spiral_xy(trait, s):
    x = s.copy()
    y = s.copy()
    last_x, last_y, last_t = 0., 0., 0.
    for i in range(s.size):
        val_x = quad(spiral_integrand_x, last_t, s[i], args=(trait.azimuth0, trait.horizontal_curvature0, trait.horizontal_curvature_gradient))[0]
        val_y = quad(spiral_integrand_y, last_t, s[i], args=(trait.azimuth0, trait.horizontal_curvature0, trait.horizontal_curvature_gradient))[0]
        last_x, last_y, last_t = last_x + val_x, last_y + val_y, s[i]
        x[i] = last_x + trait.x0
        y[i] = last_y + trait.y0
    return x, y


def line_z(trait, s):
    z = trait.z0 + s * trait.slope0
    return z


def parabola_z(trait, s):
    z = trait.z0 + s * trait.slope0 + trait.vertical_curvature * s * s / 2.
    return z


class CurveTrait(EasyDict):
    def __init__(self):
        EasyDict.__init__(self)
        self.x0 = 0.  # #0
        self.y0 = 0.  # #1
        self.z0 = 0.  # #2
        self.azimuth0 = 0.  # #3
        self.slope0 = 0.  # #4
        self.length = 0.5  # #5, the length of the parameter s in the parametric space
        self.horizontal_curvature0 = 0  # #6
        self.horizontal_curvature_gradient = 0  # #7
        self.vertical_curvature = 0  # #8


class CurveRule(ModelRule):

    def __init__(self, estimator=None):
        ModelRule.__init__(self, estimator)
        cfg = self.estimator.cfg['estimator']
        self.horizontal_type = cfg.get('horizontal_type', 'Spiral')
        self.vertical_type = cfg.get('vertical_type', 'Parabola')
        self.lower_bound = None  # lower bound as array
        self.upper_bound = None  # upper bound as array
        self.action = None        
        self.trait = None
        self.lb = None  # lower bound as trait
        self.ub = None  # upper bound as trait
        self.set_trait_range()
        self.get_xy = None
        self.get_z = None       

    def prior(self):
        lb = CurveTrait()  # lower bound
        lb.x0, lb.y0, lb.z0, lb.azimuth0, lb.slope0, lb.length = -1., -1., -1., 0.0, -0.2, 1.
        lba = np.asarray([lb.x0, lb.y0, lb.z0, lb.azimuth0, lb.slope0, lb.length])  # lower bound as array   

        ub = CurveTrait()  # upper bound
        ub.x0, ub.y0, ub.z0, ub.azimuth0, ub.slope0, ub.length = 1., 1., 1., 2.0*np.pi, 0.2, 100.
        uba = np.asarray([ub.x0, ub.y0, ub.z0, ub.azimuth0, ub.slope0, ub.length])  # upper bound as array

        example = np.asarray([0., 0., 0., 0., 0.1, 50.])
        if self.horizontal_type == 'Circle':
            lb.horizontal_curvature0 = -1./20.  # -0.05
            ub.horizontal_curvature0 = 1./20.  # 0.05
            lba = np.append(lba, lb.horizontal_curvature0)
            uba = np.append(uba, ub.horizontal_curvature0)
            example = np.append(example, 0.05)
        elif self.horizontal_type == 'Spiral':
            lb.horizontal_curvature0 = -1./20.  # -0.05
            ub.horizontal_curvature0 = 1./20.  # 0.05
            lba = np.append(lba, lb.horizontal_curvature0)
            uba = np.append(uba, ub.horizontal_curvature0)
            example = np.append(example, 0.05)

            # trait_names = np.append(trait_names, 'horizontal_curvature_gradient')
            lb.horizontal_curvature_gradient = -1./(20.*20.)  # -0.0025
            ub.horizontal_curvature_gradient = 1./(20.*20.)  # 0.0025
            lba = np.append(lba, lb.horizontal_curvature_gradient)
            uba = np.append(uba, ub.horizontal_curvature_gradient)
            example = np.append(example, 0.0025)
        else:
            assert self.horizontal_type == 'Line'

        if self.vertical_type == 'Parabola':
            lb.vertical_curvature = -1./100.  # -0.01
            ub.vertical_curvature = 1./100.  # 0.01
            lba = np.append(lba, lb.vertical_curvature)  # 0.01
            uba = np.append(uba, ub.vertical_curvature)
            example = np.append(example, 1./500.)
        else:
            assert self.vertical_type == 'Line'

        assert np.all(lba <= uba)
        # default_theta = lb+(ub-lb)/2.
        # example = ub
        return lba, uba, lb, ub, example

    def set_trait_range(self):
        # # indices of 0, 1, ..., 8 corresponding to:
        # # x0, y0, z0, azimuth0, slope0, length, horizontal_curvature0, horizontal_curvature_gradient, vertical_curvature
        # self.lower_bound = np.asarray([-1., -1., -1., 0.0, -0.2, 0., -1./20., -1./(20.*20.), -1./100.])
        # self.upper_bound = np.asarray([1., 1., 1., 2.0*np.pi, 0.2, 50., 1./20., 1./(20.*20.), 1./100.])
        prior = self.prior()
        lower_bound = prior[0]
        upper_bound = prior[1]
        lb = prior[2]
        ub = prior[3]

        min_point = self.estimator.min_point
        max_point = self.estimator.max_point
        bounding_box = max_point - min_point
        max_length = np.linalg.norm(bounding_box)
        beyond = bounding_box / 100.0  # 1/100 of the bounding box, can be adjusted        
        assert 3 == min_point.size
        lower_bound[0:3] = min_point - beyond
        lb.x0, lb.y0, lb.z0 = lower_bound[0], lower_bound[1], lower_bound[2]
        upper_bound[0:3] = max_point + beyond
        ub.x0, ub.y0, ub.z0 = upper_bound[0], upper_bound[1], upper_bound[2]

        lower_bound[5] = 2 * self.estimator.data_resolution  
        lb.length = lower_bound[5]
        upper_bound[5] = max_length
        ub.length = upper_bound[5]
        
        self.lb = lb
        self.ub = ub
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


    @staticmethod
    def measure(trait):
        return trait.length

    def compute_top_dividing_level(self):
        level = math.log2(1 + self.trait.length / self.estimator.resolution)
        assert level >= 0
        self.top_level = np.asarray([math.floor(level)])

    def generate(self):
        cloud = self.sample()
        return cloud

    def sample(self):
        trait = self.trait
        level = self.compute_current_dividing_level()[0]
        i = np.arange(1/2**(level+1), 1, 1/2**level)
        s = trait.length * i
        x, y = self.get_xy(trait, s)
        z = self.get_z(trait, s)
        cloud = np.ascontiguousarray(np.vstack((x, y, z)).transpose())
        self.estimator.add_model(new_measure=self.measure(trait), new_model=cloud)
        return cloud

    @staticmethod
    def example_traits():
        traits = []
        action = np.zeros(5)-0.1
        trait = CurveRule.parse(action=action)
        traits.append(trait)
        for _ in range(2):
            action += 0.1
            trait = CurveRule.parse(action=action)
            traits.append(trait)
        return traits
    
    def get_num_variables(self):
        assert self.lower_bound.size == self.upper_bound.size
        return self.lower_bound.size    

    # def parse(self, **kwargs):
    #     action = kwargs['action']
    #     assert action.size == self.get_num_variables()
    #     trait_flat = normalize(action, self.lower_bound, self.upper_bound)
    #     trait = CurveTrait()
    #     trait.x0 = trait_flat[0]
    #     trait.y0 = trait_flat[1]
    #     trait.z0 = trait_flat[2]
    #     trait.azimuth0 = trait_flat[3]
    #     trait.slope0 = trait_flat[4]
    #     trait.length = trait_flat[5]
    #     trait.horizontal_curvature0 = trait_flat[6]
    #     trait.horizontal_curvature_gradient = trait_flat[7]
    #     trait.vertical_curvature = trait_flat[8]
    #     self.action = action
    #     self.trait = trait
    #     self.compute_top_dividing_level()          
    #     return trait
    
    def parse(self, **kwargs):
        action = kwargs['action']
        assert action.size == self.get_num_variables()
        trait_flat = rescale(action, self.lower_bound, self.upper_bound)
        trait = CurveTrait()
        trait.x0 = trait_flat[0].astype(float)
        trait.y0 = trait_flat[1].astype(float)
        trait.z0 = trait_flat[2].astype(float)
        trait.azimuth0 = trait_flat[3].astype(float)
        trait.slope0 = trait_flat[4].astype(float)
        trait.length = trait_flat[5].astype(float)                   
        count = 6
        if self.horizontal_type == 'Line':
            self.get_xy = line_xy
        elif self.horizontal_type == 'Circle':
            trait.horizontal_curvature0 = trait_flat[count].astype(float)
            if np.isclose(trait.horizontal_curvature0, 0.0):
                self.get_xy = line_xy
            else:
                self.get_xy = circle_xy
            count = count + 1
        elif self.horizontal_type == 'Spiral':
            self.get_xy = spiral_xy
            trait.horizontal_curvature0 = trait_flat[count].astype(float)
            count = count + 1
            trait.horizontal_curvature_gradient = trait_flat[count].astype(float)
            count = count + 1
        else:
            assert False

        if self.vertical_type == 'Line':
            self.get_z = line_z
        elif self.vertical_type == 'Parabola':
            self.get_z = parabola_z
            trait.vertical_curvature = trait_flat[count].astype(float)
            count = count + 1
        else:
            assert False

        assert action.size == count
        self.action = action
        self.trait = trait        
        self.compute_top_dividing_level()        
        return trait    
