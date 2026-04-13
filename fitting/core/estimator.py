import numpy as np
import point_cloud_utils as pcu
from sklearn.neighbors import KDTree
from tools.geometry import compute_resolution
from copy import deepcopy
import open3d as o3d


# the NPRE (Nearest data Points Reguralized model-to-data Error) estimator
class Estimator:

    def __init__(self, cfg):
        self.cfg = cfg
        self.dimension = None
        self.raw_data = None
        self.data = None # np.empty((0, self.dimension), dtype=np.float32)     
        self.num_data_points = None
        self.min_point = None
        self.max_point = None
        self.data_kDTree = None
        self.data_resolution = None 
        self.model_resolution = None
        self.resolution = None # -1                                       
        self.load_data()

        self.rule = None
        self.set_rule()    

        self.regularization_factor = cfg['estimator'].get('regularization_factor', 0.5)
 
        self.current_dividing_level = -1

        self.instance_index = 0  # model instance index           

        self.model = np.empty((0, self.dimension), dtype=np.float32)
        self.labels = np.empty(0, dtype=np.int64)  # instance index labels of model points   
        self.sum_errors = 0.
        self.nearest_points = np.empty(0, dtype=np.int64)  # indexes of support points        

        self.initial_model = np.empty((0, self.dimension), dtype=np.float32)
        self.initial_labels = np.empty(0, dtype=np.int64)  # labels of model points 
        self.initial_sum_errors = 0.
        self.initial_nearest_points = np.empty(0, dtype=np.int64) # indexes of support points

        self.single_model_error = None  # model-to-data error of current single model
        self.score = None
        self.measure = 0
        self.estimator_type = cfg['estimator'].get('estimator_type', 'npre')
        self.best_models = [None,]

    def reset(self):
        self.model = deepcopy(self.initial_model)
        self.labels = deepcopy(self.initial_labels)
        self.sum_errors = deepcopy(self.initial_sum_errors)
        self.nearest_points = deepcopy(self.initial_nearest_points)
        self.measure = 0                    

    def update(self, model, sum_errors, nearest_points, labels, instance_index):
        self.initial_model = deepcopy(model)
        self.initial_labels = deepcopy(labels)
        self.initial_sum_errors = deepcopy(sum_errors)
        self.initial_nearest_points = deepcopy(nearest_points)
        self.instance_index = deepcopy(instance_index)
        assert model.shape[0] > 0
        self.best_models.append(None)
        
    def get_model(self):
        return deepcopy(self.model)
    
    def get_data(self):
        return self.raw_data
    
    def get_score(self):
        return deepcopy(self.score)
    
    def get_single_model_error(self):
        return deepcopy(self.single_model_error)
    
    def set_resolution(self, resolution):
        self.resolution = resolution

    def load_data(self):
        load_data_fn = self.cfg['estimator']['load_data_fn']
        data = load_data_fn(self)
        self.create_kdtree(data)

    # def load_data(self):
    #     if 'data_file' in self.cfg['estimator']: 
    #         load_data_fn = self.cfg['estimator']['load_data_fn']
    #         data = load_data_fn(self.cfg['estimator']['data_file'])
    #         self.dimension = data.shape[1]
    #         self.raw_data = data.copy()         
    #         self.preprocess(data, synthetic=False)
    #     else:
    #         assert False            

    def preprocess(self, data, synthetic=False):
        assert data.shape[0] > 1       
        cfg = self.cfg['estimator']

        if synthetic:
            self.data_resolution = cfg['synthetic_data_resolution']
            self.data = data
        elif 'voxel_size_for_down_sampling' in cfg:
            self.data_resolution = cfg['voxel_size_for_down_sampling']
            self.data = pcu.downsample_point_cloud_on_voxel_grid(self.data_resolution, data)
        elif 'data_resolution' in cfg:
            self.data_resolution = cfg['data_resolution']
            self.data = data
        else: 
            self.data_resolution, self.data = compute_resolution(data.copy())
        self.min_point = self.data.min(0)            
        self.max_point = self.data.max(0)        
        self.data_kDTree = KDTree(self.data)                  
        # model resolution should be smaller than 0.5 * data resolution
        self.model_resolution = cfg.get('model_resolution', 0.45 * self.data_resolution)
        assert self.model_resolution < 0.5 * self.data_resolution
        self.num_data_points = self.data.shape[0]
        self.resolution = self.model_resolution

    def create_kdtree(self, data):
        assert data.shape[0] > 1
        self.data = data
        self.dimension = data.shape[1]
        self.num_data_points = data.shape[0]     
        self.data_kDTree = KDTree(data)        

    def set_rule(self):
        rule_class = self.cfg['estimator']['rule_class']
        print(f'rule is {rule_class.__name__}')
        assert self.raw_data is not None
        self.rule = rule_class(estimator=self)

    def num_variables(self):
        assert self.rule is not None
        return self.rule.get_num_variables()

    def parse(self, **kwargs):
        trait = self.rule.parse(**kwargs)
        return trait

    def generate(self):
        assert self.rule.trait is not None
        self.rule.generate()

    def mean_measure(self):
        if self.data_kDTree is None or self.model.size == 0:
            print("no data or no model")
            return 0
        error = self.sum_errors / float(self.model.shape[0])
        if np.isclose(error, 0):
            print('the model-to-data error is impossible to be much smaller than the model resolution, please check!')
            return -1
        score = (self.measure**self.regularization_factor) / error
        return score              
        
    def npre(self):
        if self.data_kDTree is None or self.model.size == 0:
            print("no data or no model")
            return 0
        error = self.sum_errors / float(self.model.shape[0])
        if np.isclose(error, 0):
            print('the model-to-data error is impossible to be much smaller than the model resolution, please check!')
            return -1
        factor = self.regularization_factor
        # score = self.nearest_points.size / (reverse_error**factor+self.mm_epsilon)
        normalized_error = error / self.data_resolution
        normalized_regularizer = float(self.nearest_points.size) / float(self.num_data_points)
        normalized_score = (normalized_regularizer**factor) / normalized_error
        return normalized_score

    def compute_model_to_data_error(self, model):  # error from model to data
        if self.data_kDTree is None:
            print("no data")
            sum_errors = np.inf
        else:
            errors, indexes = self.data_kDTree.query(model)
            sum_errors = np.sum(errors)            
            self.nearest_points = np.unique(np.concatenate((self.nearest_points, indexes[:,0])))                   
        return sum_errors

    def add_model(self, **kwargs): 
        new_model = kwargs['new_model']        
        if isinstance(new_model, o3d.core.Tensor):
            new_model = new_model.cpu().numpy()
        elif isinstance(new_model, o3d.geometry.PointCloud):
            new_model = np.asarray(new_model.points)
        if 2 == self.dimension and 3 == new_model.shape[1]:
            new_model = new_model[:,:2]
        if new_model.shape[0] == 0:
            print('error: new model has no points')
            assert False
        if new_model.shape[0] < 5:                             
            if self.current_dividing_level != 0:  # not early rejection
                # print('warning: new model is too small')
                self.score, self.single_model_error = -1, float('inf')
                return
            else:
                #  print('early rejection')
                pass
        self.best_models[-1] = new_model
        self.measure += kwargs.get('new_measure', new_model.shape[0])
        sum_errors = self.compute_model_to_data_error(new_model)
        self.single_model_error = sum_errors / float(new_model.shape[0])

        self.sum_errors += sum_errors         
        self.model = np.vstack((self.model, new_model))
        new_labels = np.full(new_model.shape[0], self.instance_index)
        self.labels = np.concatenate((self.labels, new_labels))

        if self.estimator_type == 'npre':
            self.score = self.npre()
        elif self.estimator_type == 'mean measure': 
            self.score = self.mean_measure()
        else:
            assert False


                 