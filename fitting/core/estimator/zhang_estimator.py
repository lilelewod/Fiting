import numpy as np
import point_cloud_utils as pcu
import time
from sklearn.neighbors import KDTree
from tools.geometry import compute_resolution
from copy import deepcopy
from models.rule import Token


try:
    from superquadric_fast import model_to_data_error_bruteforce as _fast_model_to_data_error_bruteforce
except Exception:  # pragma: no cover - optional extension
    _fast_model_to_data_error_bruteforce = None

try:
    import faiss
except Exception:  # pragma: no cover - optional dependency
    faiss = None


# the NPRE (Nearest data Points Reguralized model-to-data Error) estimator
class Estimator:  # estimate one solution

    def __init__(self, cfg, faiss_lock=None):
        self.cfg = cfg
        self._faiss_lock = faiss_lock
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
 
        self.current_is_early_rejection = False

        self.index = 0  # model instance index           

        self.token = None  # np.empty((0, self.dimension), dtype=np.float32)
        # self.labels = np.empty(0, dtype=np.int64)  # instance index labels of model points   
        self.sum_errors = 0.
        self.supporters = np.empty(0, dtype=np.int64)  # indexes of nearest data points
        self.num_points = 0        

        self.base_sum_errors = 0.
        self.base_supporters = np.empty(0, dtype=np.int64) # indexes of support points
        self.base_num_points = 0

        self.single_model_error = None  # model-to-data error of current single model
        self.score = None
        self.measure = 0.
        self.estimator_type = cfg['estimator'].get('estimator_type', 'npre')
        self.score_npre = 0  # NPRE: Nearest data Points Reguralized model-to-data Error
        self.score_mm = 0  # MM: mean measure
        self.last_surface_sampling_time = 0.0
        self.last_model_to_data_error_time = 0.0
        self.use_fast_bruteforce_model_to_data = cfg['estimator'].get(
            'use_fast_bruteforce_model_to_data', False)
        self.model_to_data_backend = cfg['estimator'].get('model_to_data_backend', None)
        if self.model_to_data_backend is None:
            self.model_to_data_backend = (
                'fast_bruteforce' if self.use_fast_bruteforce_model_to_data else 'sklearn'
            )
        self.faiss_gpu_device = int(cfg['estimator'].get('faiss_gpu_device', 0))
        self.faiss_gpu_use_float16 = bool(cfg['estimator'].get('faiss_gpu_use_float16', False))
        self.faiss_gpu_min_query_points = int(cfg['estimator'].get('faiss_gpu_min_query_points', 32))
        self.faiss_gpu_resources = None
        self.faiss_index = None
        self._faiss_unavailable_reported = False

        self.sampling_level = 0

    def reset(self):
        self.sum_errors = deepcopy(self.base_sum_errors)
        self.supporters = deepcopy(self.base_supporters)
        self.num_points = deepcopy(self.base_num_points)
        self.measure = 0.                    

    def update(self, supporters, sum_errors, num_points):
        self.base_sum_errors = deepcopy(sum_errors)
        self.base_supporters = deepcopy(supporters)
        self.base_num_points = deepcopy(num_points)
        
    def get_token(self):
        return deepcopy(self.token)
    
    def get_data(self):
        return self.data
    
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
        self.faiss_gpu_resources = None
        self.faiss_index = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_faiss_lock'] = None
        state['faiss_gpu_resources'] = None
        state['faiss_index'] = None
        return state

    def set_rule(self):
        rule_class = self.cfg['estimator'].get('rule_class', None)
        if rule_class is None:
            return
        print(f'rule is {rule_class.__name__}')
        assert self.raw_data is not None
        self.rule = rule_class(estimator=self)

    def num_variables(self):
        assert self.rule is not None
        return self.rule.get_num_variables()

    def parse(self, **kwargs):
        trait = self.rule.parse(**kwargs)
        return trait

    def clear_current_candidate(self):
        self.token = None
        self.score = -1
        self.score_npre = -1
        self.score_mm = -1
        self.single_model_error = float('inf')

    def generate(self, sampling_level=0):
        self.sampling_level = sampling_level
        resolution = self.model_resolution * (2.0 ** sampling_level)
        self.set_resolution(float(resolution))

        self.last_surface_sampling_time = 0.0
        self.last_model_to_data_error_time = 0.0
        self.clear_current_candidate()
        assert self.rule.trait is not None
        self.rule.generate()

    def prepare_model_to_data_backend(self):
        if self.model_to_data_backend == 'faiss_gpu':
            self._ensure_faiss_gpu_index()

    def estimate(self):
        if self.data_kDTree is None or self.token is None or self.token.points.size == 0:
            print("no data or no model")
            return 0
        error = self.sum_errors / float(self.num_points)
        if np.isclose(error, 0):
            print('the model-to-data error is impossible to be much smaller than the model resolution, please check!')
            return -1
        factor = self.regularization_factor
        # score = self.nearest_points.size / (reverse_error**factor+self.mm_epsilon)
        normalized_error = error / self.data_resolution
        normalized_regularizer = float(self.supporters.size) / float(self.num_data_points)
        self.score_npre = (normalized_regularizer**factor) / normalized_error
        self.score_mm = (self.measure**factor) / normalized_error
        
        if self.estimator_type == 'npre':
            self.score = self.score_npre
        elif self.estimator_type == 'mean measure':
            self.score = self.score_mm
        else:
            assert False

    def compute_model_to_data_error(self, points):  # error from model to data
        if self.data_kDTree is None:
            print("no data")
            sum_errors = np.inf
        else:
            start = time.perf_counter()
            use_faiss = (
                self.model_to_data_backend == 'faiss_gpu'
                and points.shape[0] >= self.faiss_gpu_min_query_points
                and self._ensure_faiss_gpu_index()
            )
            if use_faiss:
                if self._faiss_lock is not None:
                    self._faiss_lock.acquire()
                try:
                    distances_sq, indexes = self.faiss_index.search(
                        np.ascontiguousarray(points, dtype=np.float32),
                        1,
                    )
                finally:
                    if self._faiss_lock is not None:
                        self._faiss_lock.release()
                distances = np.sqrt(np.maximum(distances_sq[:, 0], 0.0))
                indexes = np.asarray(indexes, dtype=np.int64)
                sum_errors = float(np.sum(distances))
                self.supporters = np.unique(np.concatenate((self.supporters, indexes[:, 0])))
            elif (
                self.model_to_data_backend in ('fast_bruteforce', 'faiss_gpu')
                and _fast_model_to_data_error_bruteforce is not None
            ):
                sum_errors, supporters, merged_supporters = _fast_model_to_data_error_bruteforce(
                    np.ascontiguousarray(points, dtype=np.float32),
                    np.ascontiguousarray(self.data, dtype=np.float32),
                    np.asarray(self.supporters, dtype=np.int64),
                )
                indexes = np.asarray(supporters, dtype=np.int64)[:, None]
                self.supporters = np.asarray(merged_supporters, dtype=np.int64)
            else:
                errors, indexes = self.data_kDTree.query(points)
                sum_errors = np.sum(errors)
                self.supporters = np.unique(np.concatenate((self.supporters, indexes[:,0])))
            self.last_model_to_data_error_time += time.perf_counter() - start
        return sum_errors, indexes[:,0]

    def _ensure_faiss_gpu_index(self):
        if self.faiss_index is not None:
            return True
        if faiss is None or not hasattr(faiss, 'get_num_gpus') or faiss.get_num_gpus() <= 0:
            if not self._faiss_unavailable_reported:
                print('faiss-gpu is unavailable; falling back to sklearn KDTree')
                self._faiss_unavailable_reported = True
            return False
        data = np.ascontiguousarray(self.data, dtype=np.float32)
        cpu_index = faiss.IndexFlatL2(data.shape[1])
        self.faiss_gpu_resources = faiss.StandardGpuResources()
        options = faiss.GpuClonerOptions()
        options.useFloat16 = self.faiss_gpu_use_float16
        self.faiss_index = faiss.index_cpu_to_gpu(
            self.faiss_gpu_resources,
            self.faiss_gpu_device,
            cpu_index,
            options,
        )
        self.faiss_index.add(data)
        return True

    def add_token(self, token: Token):
        points = token.points       
        if 2 == self.dimension and 3 == points.shape[1]:
            points = points[:,:2]
        if points.shape[0] == 0:
            print('error: the new model instance has no points')
            assert False
        if points.shape[0] < 5 and self.sampling_level == 0:  # model point clouds has very few points (<5), 5 can be adjusted 
            # print('warning: new model is too small')
            self.clear_current_candidate()
            return

        sum_errors, supporters = self.compute_model_to_data_error(points)
        self.single_model_error = sum_errors / float(points.shape[0])

        token.supporters = supporters
        token.sum_errors = sum_errors

        self.sum_errors += sum_errors
        self.num_points += points.shape[0]
        self.measure += token.measure    
        self.token = token

        self.estimate()     
