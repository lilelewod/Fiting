import time
from copy import deepcopy
import os
import pickle
from tools.time_tool import current_timestamp
from tools.plot_manager import PlotManager
import json
from easydict import EasyDict
from tools.tool import json_default
from tools.geometry import save_point_cloud


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
        self.best_model = None
        self.best_estimator = None
        self.colors = []
        self.times = []
        self.data_cloud = None
        self.name = env_id
        self.ideal_score = None
        self.best_single_model_error = float('inf')

    def update(self, score, env, trait=None, action=None):
        self.total_episode += 1
        self.episode += 1
        if score >= self.best_score:
            estimator = env.estimator
            self.best_score = score
            self.scores.append(score)
            self.total_episodes.append(self.total_episode)
            self.episodes.append(self.episode)
            self.traits.append(trait)
            self.actions.append(action)
            self.best_model = estimator.get_model()
            self.best_single_model_error = deepcopy(
                estimator.single_model_error)
            self.best_estimator = deepcopy(estimator)
            self.colors.append(deepcopy(estimator.model_color))
            self.times.append(self.clock() - self.start_time)
            return True
        return False


class Record(EasyDict):
    def __init__(self, cfg, dimension=3):
        self.cfg = cfg
        self.log_dir = None
        # self.out_file_name = None
        self.out_json_file_name = None        

        self.timestamp = cfg['record'].get('timestamp') or current_timestamp()
        self.name = cfg.get('record_name', -1)
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

        self.data_cloud = None
        self.best_score = 0.
        self.best_model = None
        self.best_estimator = None
        self.best_sub_record = -1
        self.evolving_scores = []
        self.evolving_episodes = []
        self.evolving_times = []
        self.evolving_actions = []
        self.evolving_traits = []

    def close(self):
        self.plotter.close()

    def make_log_dir(self):
        root_dir = self.cfg['record'].get('root_dir', './ouputs/')
        log_dir = root_dir + self.timestamp + '/'
        self.log_dir = log_dir if self.name == -1 else log_dir + self.name + '/'

        os.makedirs(self.log_dir, mode=0o777, exist_ok=True)
        # self.out_file_name = self.log_dir + 'induction_log.txt'
        self.out_json_file_name = self.log_dir + 'record.json'

        # with open(self.out_file_name, "a+") as file:
        #     file.write(
        #         f'induction configurations:\nlog dir= {self.log_dir}\n\n')

    def save_to_file(self):
        results = {'cfg': self.cfg, 'evolved_time': self.evolution['evolved_time'],
                   'evolved_scores': self.evolution['evolved_scores'],
                   'evolved_iterations': self.evolution['evolved_iterations']}
        with open(self.log_dir + '/' + 'induction.pckl', 'wb') as f:
            pickle.dump(results, f)
        print('induction is completed')

    def update(self, record: SubRecord, elapsed_episodes, **kwargs):
        self.update_count += 1
        self.episode += elapsed_episodes
        if record is None:
            if self.update_count % self.pulse_size == 0:
                model_size = 0 if self.best_model is None else self.best_model.shape[0]
                print(f'record {self.name}: after {self.episode} episodes the best score still is {self.best_score:.4f}, '
                      f'model size is {model_size}, got by sub-record {self.best_sub_record}')
                policy = kwargs.get('policy', None)
                if policy is not None:
                    print(f'\npolicy mean {policy.mean}')
                    print(f'policy log std {policy.log_std}')                      
            return False
        score = record.best_score
        if score > self.best_score:
            self.best_score = score
            self.best_sub_record = record.name
            best_episode = self.episode - \
                (elapsed_episodes - record.episodes[-1])
            self.evolving_scores.append(score)
            self.evolving_episodes.append(best_episode)
            self.evolving_actions.append(record.actions[-1])
            self.evolving_traits.append(record.traits[-1])
            self.evolving_times.append(self.clock() - self.start_time)

            with open(self.out_json_file_name, 'w') as out_file:
                # json.dump(self, out_file, default=lambda o: '<not serializable>', indent=2)
                json.dump(self, out_file, default=json_default, indent=2)
            self.best_model = record.best_model
            self.best_estimator = record.best_estimator

            model_color = record.colors[-1]
            model_image = None
            print(f'record {self.name}: after {best_episode} episodes the best score becomes '
                  f'{self.best_score:.4f}, model size is {self.best_model.shape[0]}, got by sub-record {self.best_sub_record}')
            self.plotter.plot(runner_id=self.name, rollout_id=self.best_sub_record, episodes=self.evolving_episodes, scores=self.evolving_scores,
                              times=self.evolving_times, model=self.best_model, data=self.data_cloud, model_color=model_color, model_image=model_image)
            save_point_cloud(self.best_model, self.log_dir+'best_model.ply')
            return True
        return False
