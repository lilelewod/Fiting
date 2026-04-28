import time
from copy import deepcopy, copy
import os
import pickle
from tools.time_tool import current_timestamp
from tools.plot_manager import PlotManager
import json
from easydict import EasyDict
from tools.tool import json_default
from tools.geometry import save_point_cloud, save_triangle_mesh
import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial import Delaunay, QhullError


class SubRecord:
    def __init__(self, cfg, env_id=0):
        self.cfg = cfg
        self.clock = time.thread_time if cfg['record'].get(
            'use_thread_time', False) else time.time
        self.start_time = self.clock()

        self.total_episode = 0
        self.episode = 0
        self.best_score = 0.
        self.scores = []
        self.total_episodes = []
        self.episodes = []
        self.traits = []
        self.actions = []
        self.best_token = None
        self.best_estimator = None
        self.times = []
        self.data_cloud = None
        self.name = env_id
        self.ideal_score = None
        self.best_single_model_error = float('inf')
        self.best_score_mm = 0.

    def update(self, score, estimator):
        self.total_episode += 1
        self.episode += 1
        if estimator.score_mm > self.best_score_mm:
            self.best_score_mm = estimator.score_mm
        if score >= self.best_score:
            self.best_score = score
            self.scores.append(score)
            self.total_episodes.append(self.total_episode)
            self.episodes.append(self.episode)
            self.best_token = estimator.get_token()
            self.traits.append(self.best_token.trait)
            self.actions.append(self.best_token.action)            
            self.best_single_model_error = deepcopy(
                estimator.single_model_error)
            self.best_estimator = deepcopy(estimator)
            self.times.append(self.clock() - self.start_time)
            return True
        return False


class Record(EasyDict):
    def __init__(self, cfg, dimension=3):
        self.cfg = cfg
        self.dimension = dimension
        self.lower_bound = None
        self.upper_bound = None        
        self.log_dir = None
        self.out_json_file_name = None        

        self.timestamp = cfg['record'].get('timestamp') or current_timestamp()
        self.name = cfg['record'].get('name', -1)
        self.verbose = cfg['record'].get('verbose', True)
        self.clock = time.thread_time if cfg['record'].get(
            'use_thread_time', False) else time.time
        self.start_time = self.clock()
        self.make_log_dir()
        
        visualization = cfg['record'].get('visualization', None)
        rotate = cfg['record'].get('rotate', False)
        self.plotter = PlotManager(visualization=visualization, compared_file=cfg.get(
            'compared_file', None), dimension=dimension, log_dir=self.log_dir, rotate=rotate)
        self.pulse_size = cfg['record'].get('pulse_size', 1000)
        self.update_count = 0
        self.episode = 0
        self.total_episode = 0

        self.token_index = 0
        self.data_cloud = None
        self.best_score = 0.
        self.best_cloud = None
        self.best_estimator = None
        self.best_token_set = [None, ] * cfg['fitter']['num_instances']
        self.best_color = None
        self.best_sub_record = -1
        self.evolving_scores = []
        self.evolving_episodes = []
        self.evolving_times = []
        self.evolving_actions = []
        self.evolving_traits = []
        self.evolutions = dict()
        self.base_cloud = None
        self.base_color = None
        self.round = None

    @staticmethod
    def _grid_faces(rows, cols, offset=0):
        faces = []
        for row in range(rows - 1):
            for col in range(cols - 1):
                v00 = offset + row * cols + col
                v01 = v00 + 1
                v10 = v00 + cols
                v11 = v10 + 1
                faces.append([v00, v10, v01])
                faces.append([v01, v10, v11])
        return np.asarray(faces, dtype=np.int32)

    def _save_merged_mesh(self):
        model_cfg = self.cfg.get('model', {})
        sample_u = int(model_cfg.get('sample_u', 0))
        sample_v = int(model_cfg.get('sample_v', 0))
        expected_points = sample_u * sample_v
        if expected_points <= 0:
            return

        vertices = []
        faces = []
        offset = 0
        for token in self.best_token_set:
            if token is None or getattr(token, 'points', None) is None:
                continue
            points = np.asarray(token.points)
            if points.shape[0] != expected_points:
                return
            vertices.append(points)
            faces.append(self._grid_faces(sample_u, sample_v, offset=offset))
            offset += points.shape[0]

        if not vertices:
            return

        vertices = np.vstack(vertices)
        faces = np.vstack(faces)
        save_triangle_mesh(vertices, faces, self.log_dir + 'final_merged_mesh.ply')
        self._save_trimmed_merged_mesh(vertices, faces)
        self._save_uv_trimmed_merged_mesh(sample_u, sample_v)

    def _save_trimmed_merged_mesh(self, vertices, faces):
        record_cfg = self.cfg.get('record', {})
        estimator_cfg = self.cfg.get('estimator', {})
        if not record_cfg.get('trim_final_mesh', False):
            return
        if self.data_cloud is None or len(vertices) == 0 or len(faces) == 0:
            return

        data = np.asarray(self.data_cloud)
        data_resolution = float(estimator_cfg.get('data_resolution', 0.0))
        if data_resolution <= 0.0:
            return

        distance_factor = float(record_cfg.get('trim_distance_factor', 2.5))
        max_distance = distance_factor * data_resolution
        distances = KDTree(data).query(vertices, k=1, return_distance=True)[0].reshape(-1)
        keep_vertices = distances <= max_distance

        bbox_margin_factor = float(record_cfg.get('trim_bbox_margin_factor', 1.0))
        if bbox_margin_factor > 0.0:
            margin = bbox_margin_factor * data_resolution
            min_point = np.min(data, axis=0) - margin
            max_point = np.max(data, axis=0) + margin
            inside_bbox = np.all((vertices >= min_point) & (vertices <= max_point), axis=1)
            keep_vertices &= inside_bbox

        keep_faces = np.all(keep_vertices[faces], axis=1)
        trimmed_faces = faces[keep_faces]
        if trimmed_faces.size == 0:
            return

        used_vertices = np.unique(trimmed_faces.reshape(-1))
        remap = np.full(vertices.shape[0], -1, dtype=np.int32)
        remap[used_vertices] = np.arange(used_vertices.size, dtype=np.int32)
        trimmed_vertices = vertices[used_vertices]
        trimmed_faces = remap[trimmed_faces]

        save_triangle_mesh(
            trimmed_vertices,
            trimmed_faces,
            self.log_dir + 'final_merged_mesh_trimmed.ply',
        )

    @staticmethod
    def _uv_grid(rows, cols):
        u = np.linspace(0.0, 1.0, rows, dtype=np.float32)
        v = np.linspace(0.0, 1.0, cols, dtype=np.float32)
        uu, vv = np.meshgrid(u, v, indexing='ij')
        return np.stack((uu, vv), axis=-1).reshape(-1, 2)

    @staticmethod
    def _compact_mesh(vertices, faces):
        if faces.size == 0:
            return None, None
        used_vertices = np.unique(faces.reshape(-1))
        remap = np.full(vertices.shape[0], -1, dtype=np.int32)
        remap[used_vertices] = np.arange(used_vertices.size, dtype=np.int32)
        return vertices[used_vertices], remap[faces]

    def _uv_trim_faces(self, points, sample_u, sample_v, tree, data, data_resolution):
        record_cfg = self.cfg.get('record', {})
        distance_factor = float(record_cfg.get('uv_trim_distance_factor', 2.5))
        max_distance = distance_factor * data_resolution
        distances = tree.query(points, k=1, return_distance=True)[0].reshape(-1)
        inlier_mask = distances <= max_distance

        bbox_margin_factor = float(record_cfg.get('uv_trim_bbox_margin_factor', 1.0))
        if bbox_margin_factor > 0.0:
            margin = bbox_margin_factor * data_resolution
            min_point = np.min(data, axis=0) - margin
            max_point = np.max(data, axis=0) + margin
            inlier_mask &= np.all((points >= min_point) & (points <= max_point), axis=1)

        local_faces = self._grid_faces(sample_u, sample_v)
        uv = self._uv_grid(sample_u, sample_v)
        inlier_uv = uv[inlier_mask]
        if inlier_uv.shape[0] < 3:
            return local_faces[np.all(inlier_mask[local_faces], axis=1)]

        try:
            delaunay = Delaunay(inlier_uv)
        except QhullError:
            return local_faces[np.all(inlier_mask[local_faces], axis=1)]

        if inlier_uv.shape[0] >= 4:
            uv_tree = KDTree(inlier_uv)
            nearest = uv_tree.query(inlier_uv, k=2, return_distance=True)[0][:, 1]
            edge_limit = float(record_cfg.get('uv_trim_edge_factor', 4.0)) * max(
                float(np.median(nearest)), np.finfo(np.float32).eps
            )
            tri_uv = inlier_uv[delaunay.simplices]
            edge_lengths = np.stack(
                (
                    np.linalg.norm(tri_uv[:, 0] - tri_uv[:, 1], axis=1),
                    np.linalg.norm(tri_uv[:, 1] - tri_uv[:, 2], axis=1),
                    np.linalg.norm(tri_uv[:, 2] - tri_uv[:, 0], axis=1),
                ),
                axis=1,
            )
            kept_simplices = np.max(edge_lengths, axis=1) <= edge_limit
            if not np.any(kept_simplices):
                kept_simplices = np.ones(delaunay.simplices.shape[0], dtype=bool)
        else:
            kept_simplices = np.ones(delaunay.simplices.shape[0], dtype=bool)

        face_centers = np.mean(uv[local_faces], axis=1)
        simplex_ids = delaunay.find_simplex(face_centers)
        keep_faces = simplex_ids >= 0
        keep_faces[keep_faces] &= kept_simplices[simplex_ids[keep_faces]]
        return local_faces[keep_faces]

    def _save_uv_trimmed_merged_mesh(self, sample_u, sample_v):
        record_cfg = self.cfg.get('record', {})
        estimator_cfg = self.cfg.get('estimator', {})
        if not record_cfg.get('uv_trim_final_mesh', False):
            return
        if self.data_cloud is None:
            return

        data_resolution = float(estimator_cfg.get('data_resolution', 0.0))
        if data_resolution <= 0.0:
            return

        data = np.asarray(self.data_cloud)
        tree = KDTree(data)
        vertices = []
        faces = []
        offset = 0
        expected_points = sample_u * sample_v
        for token in self.best_token_set:
            if token is None or getattr(token, 'points', None) is None:
                continue
            points = np.asarray(token.points)
            if points.shape[0] != expected_points:
                return
            local_faces = self._uv_trim_faces(points, sample_u, sample_v, tree, data, data_resolution)
            if local_faces.size == 0:
                continue
            vertices.append(points)
            faces.append(local_faces + offset)
            offset += points.shape[0]

        if not vertices or not faces:
            return

        vertices = np.vstack(vertices)
        faces = np.vstack(faces)
        vertices, faces = self._compact_mesh(vertices, faces)
        if vertices is None:
            return
        save_triangle_mesh(
            vertices,
            faces,
            self.log_dir + 'final_merged_mesh_uv_trimmed.ply',
        )

    def close(self):
        self.plotter.close()

    def make_log_dir(self):
        root_dir = self.cfg['record'].get('root_dir', 'ouputs')
        log_dir = root_dir + '/' + self.timestamp + '/'
        self.log_dir = log_dir if self.name == -1 else log_dir + self.name + '/'

        os.makedirs(self.log_dir, mode=0o777, exist_ok=True)
        self.out_json_file_name = self.log_dir + 'record.json'

    def save_to_file(self):
        results = {'cfg': self.cfg, 'evolved_time': self.evolution['evolved_time'],
                   'evolved_scores': self.evolution['evolved_scores'],
                   'evolved_iterations': self.evolution['evolved_iterations']}
        with open(self.log_dir + '/' + 'induction.pckl', 'wb') as f:
            pickle.dump(results, f)
        print('induction is completed')

    def get_base(self):
        supporters = np.empty(0, dtype=np.int64)  # Indices of the nearest data points for each model point
        sum_errors = 0
        num_points = 0
        self.base_cloud = np.empty((0, self.dimension), dtype=np.float32)
        self.base_color = np.empty(0, dtype=np.int64)
        for k in range(self.cfg['fitter']['num_instances']):
            token = self.best_token_set[k]
            if k == self.token_index or token is None:
                continue
            self.base_cloud = np.vstack((self.base_cloud, token.points))
            self.base_color = np.concatenate((self.base_color, token.color))
            supporters = np.unique(np.concatenate((supporters, token.supporters)))
            sum_errors += token.sum_errors
            num_points += token.points.shape[0]
        return supporters, sum_errors, num_points      

    def update(self, record: SubRecord, elapsed_episodes, **kwargs):
        self.update_count += 1
        self.episode += elapsed_episodes
        self.total_episode += elapsed_episodes
        if record is None:
            if (self.update_count % self.pulse_size == 0) and self.verbose:
                model_size = 0 if self.best_cloud is None else self.best_cloud.shape[0]
                print(f'record {self.name}: after {self.episode} episodes the best score still is {self.best_score:.4f}, '
                      f'model size is {model_size}, got by sub-record {self.best_sub_record}')
                policy = kwargs.get('policy', None)
                if policy is not None:
                    print(f'\npolicy mean is {policy.get_mean()}')
                    print(f'policy standard deviation is {policy.get_std()}')                      
            return False
        score = record.best_score
        if score > self.best_score:
            self.best_score = score
            self.best_sub_record = record.name
            best_episode = self.episode - (elapsed_episodes - record.episodes[-1])
            elpased_time = self.clock() - self.start_time
            if self.base_cloud is None or self.base_color is None:
                self.get_base()
            if self.round is None:
                self.round = 0
            evolution_key = f'round_{self.round}_instance_{self.token_index}'
            if evolution_key not in self.evolutions:
                self.evolutions[evolution_key] = []
            self.evolving_scores.append(score)
            self.evolving_episodes.append(best_episode)
            # self.evolving_actions.append(record.actions[-1])
            # self.evolving_traits.append(record.traits[-1])
            self.evolving_times.append(elpased_time)
            best = dict(
                episode=best_episode,
                elpased_time=elpased_time,
                score=score,
                trait=record.traits[-1],
            )            
            # easydict 需要用string 作为key
            self.evolutions[evolution_key].append(best)

            with open(self.out_json_file_name, 'w') as out_file:
                log = copy(self)
                log.plotter = None
                log.evolutions = 'see the files named evolution_round_{}_of_instance_{}.json'                
                json.dump(log, out_file, default=json_default, indent=2)
            with open(f'{self.log_dir}evolution_of_round_{self.round}_instance_{self.token_index}.json', 'w') as out_file:
                json.dump(self.evolutions[evolution_key], out_file, default=json_default, indent=2)

            token = record.best_token
            self.best_estimator = deepcopy(record.best_estimator)
            token.color = np.full(token.points.shape[0], self.token_index)
            self.best_token_set[self.token_index] = token

            self.best_cloud = np.vstack((self.base_cloud, token.points))
            self.best_color = np.concatenate((self.base_color, token.color))            

            model_image = None
            if self.verbose:
                print(f'record {self.name}: after {best_episode} episodes the best score becomes '
                    f'{self.best_score:.4f}, model size is {self.best_cloud.shape[0]}, got by sub-record {self.best_sub_record}')
            self.plotter.plot(runner_id=self.name, rollout_id=self.best_sub_record, episodes=self.evolving_episodes, scores=self.evolving_scores,
                              times=self.evolving_times, model=self.best_cloud, data=self.data_cloud, model_labels=self.best_color, model_image=model_image)
            save_point_cloud(token.points, self.log_dir + f'best_cloud_of_instance_{self.token_index}.ply')
            if self.cfg['fitter']['num_instances'] > 1:
                save_point_cloud(self.best_cloud, self.log_dir + 'final_merged.ply')
                self._save_merged_mesh()
            return True
        return False
