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
