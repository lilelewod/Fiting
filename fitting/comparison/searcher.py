import numpy as np
import setproctitle
from copy import deepcopy
import math
from tools.tool import set_seed, get_seeds, init_device
from core.record import Record
import pickle


def simple_bounds(s, lb, ub):
    index = s < lb
    s[index] = lb[index]
    index = s > ub
    s[index] = ub[index]
    return s


def get_cuckoos(nest, best, lb, ub):
    new_nest = deepcopy(nest)
    n = nest.shape[0]
    beta = 3/2
    sigma = (math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    for i in range(n):
        s = nest[i, :]
        u = np.random.standard_normal(s.shape)*sigma
        v = np.random.standard_normal(s.shape)
        step = u/np.abs(v)**(1/beta)
        step_size = 0.01*step*(s-best)
        s = s+step_size*np.random.standard_normal(s.shape)
        new_nest[i, :] = simple_bounds(s, lb, ub)
    return new_nest


def empty_nests(nest, lb, ub, pa):
    n = nest.shape[0]
    k = np.random.random_sample(nest.shape) > pa
    step_size = np.random.random_sample()*(nest[np.random.permutation(n), :]-nest[np.random.permutation(n), :])
    new_nest = nest + step_size*k
    for i in range(n):
        s = new_nest[i, :]
        new_nest[i, :] = simple_bounds(s, lb, ub)
    return new_nest


class Searcher:  # cuckoo searcher

    def __init__(self, cfg):
        self.cfg = cfg
        self.num_envs = int(cfg['searcher']['num_envs'])
        seeds = self.cfg.get('seeds', None)
        if seeds is None:
            seeds = get_seeds(self.num_envs+1)
            self.cfg['seeds'] = seeds
            self.cfg['raw_seeds'] = None
        set_seed(seeds[-1])

        self.device = init_device(cfg['device'])
        cfg['raw_device'] = deepcopy(cfg['device'])
        cfg['device'] = self.device

        # set the title of the process
        setproctitle.setproctitle(cfg['estimator']['data_file'] + '-' + cfg['searcher']['name'])

        self.population_size = cfg['searcher']['episodes_per_env'] * self.num_envs
        if cfg['collector']['parallel']:
            from core.collector import Collector
        else:
            assert False
            from .runner import SerialRunner as Runner
        self.collector = Collector(cfg, self.num_envs)
        data_cloud = self.collector.launch()
        self.record = Record(cfg, dimension=data_cloud.shape[1])
        self.record.data_cloud = data_cloud

    def estimate(self, solutions):
        assert solutions.shape[0] == self.population_size
        episodes = self.cfg['searcher']['episodes_per_env']
        scores = np.zeros(self.population_size)
        for env_id in range(self.num_envs):
            actions = solutions[env_id*episodes:(env_id+1)*episodes]
            self.collector.estimate(env_id=env_id, actions=actions)
        for env_id in range(self.num_envs):
            try:
                scores[env_id*episodes:(env_id+1)*episodes], record = self.collector.receive(env_id)
                self.record.update(record, episodes)                
            except pickle.UnpicklingError as e:
                assert False 
        return scores

    def search(self):
        dim = self.collector.get_action_dim()
        lower_bound = np.full(dim, -1.)
        upper_bound = np.full(dim, 1.)        
        n = self.population_size
        pa = 0.25
        nest = np.zeros((n, dim), dtype=np.float32)
        for i in range(n):
            nest[i, :] = lower_bound + (upper_bound - lower_bound) * np.random.random_sample(dim)

        max_iteration = self.cfg['searcher']['max_episode']
        iteration = 0
        while iteration < max_iteration:
            scores = self.estimate(solutions=nest)
            iteration += n
            best_nest = nest[np.argmax(scores), :]         
            nest = get_cuckoos(nest, best_nest, lower_bound, upper_bound)
                        
            scores = self.estimate(solutions=nest)                        
            nest = empty_nests(nest, lower_bound, upper_bound, pa)
            iteration += n
        
        print('the search is finished')
    
    def close(self):
        self.collector.close()
        self.record.close()        
        
