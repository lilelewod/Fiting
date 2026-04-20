from typing import Any
from multiprocessing import get_context
# from ditk import logging
import time
import traceback
import torch
import pickle
import cloudpickle
import numpy as np
from copy import deepcopy
from .environment import Environment as Env


def get_env_fn(cfg, estimator, env_id):
    def init_env():
        env = Env(cfg=cfg, estimator=estimator, env_id=env_id)
        return env
    return init_env


class CloudPickleWrapper:
    """
    Overview:
        CloudPickleWrapper can be able to pickle more python object (e.g., an object with lambda expression)
    """

    def __init__(self, data: Any) -> None:
        self.data = data

    def __getstate__(self) -> bytes:
        return cloudpickle.dumps(self.data)

    def __setstate__(self, data: bytes) -> None:
        if isinstance(data, (tuple, list, np.ndarray)):  # pickle is faster
            self.data = pickle.loads(data)
        else:
            self.data = cloudpickle.loads(data)


class Collector:
    """
    Overview:
        Create an SubprocessCollector to manage multiple envs.
        Each env is run by a respective subprocess.
    """

    def __init__(self, cfg, num_envs):
        self.cfg = cfg
        self.num_envs = num_envs   
        self.env_fns = None

        self.estimator = self.cfg['estimator']['estimator_class'](self.cfg)         
     
        self.action_dim = self.estimator.num_variables()             
       
    def get_action_dim(self):
        return self.action_dim     
  
    def _create_state(self) -> None:
        r"""
        Overview:
            Fork/spawn sub-processes(Call ``_create_env_subprocess``) and create pipes to transfer the data.
        """
        self._pipe_parents, self._pipe_children = {}, {}
        self._subprocesses = {}
        for env_id in range(self.num_envs):
            self._create_env_subprocess(env_id)         
        self._waiting_env = {'step': set()}
        self._closed = False

    def _create_env_subprocess(self, env_id):
        ctx = get_context('spawn')
        self._pipe_parents[env_id], self._pipe_children[env_id] = ctx.Pipe()
        self._subprocesses[env_id] = ctx.Process(
            target=self.worker_fn_robust,
            args=(
                self._pipe_parents[env_id],
                self._pipe_children[env_id],
                CloudPickleWrapper(self.env_fns[env_id]),              
                self.method_name_list,
            ),
            daemon=True,
            name='subprocess_runner{}_{}'.format(env_id, time.time())
        )
        self._subprocesses[env_id].start()
        self._pipe_children[env_id].close()

    @property
    def method_name_list(self) -> list:
        return ['reset', 'seed', 'close', 'set_seed', 'launch', 'estimate', 'update']

    @staticmethod
    def worker_fn_robust(
            parent,
            child,
            env_fn_wrapper,            
            method_name_list,
    ) -> None:
        torch.set_num_threads(1)  # TODO: could be tunned
        env_fn = env_fn_wrapper.data
        env = env_fn()                              
        parent.close()
       
        def set_seed_fn(*args, **kwargs):
            timestep = env.set_seed(*args, **kwargs)
            return timestep
        
        def launch_fn(*args, **kwargs):
            env.launch(*args, **kwargs)

        def reset_fn(*args, **kwargs):
            obs_n = env.reset(*args, **kwargs)
            return obs_n

        def estimate_fn(*args, **kwargs):
            return env.estimate(*args, **kwargs)

        def update_fn(*args, **kwargs):
            return env.update(*args, **kwargs)                        

        while True:
            try:
                cmd, args, kwargs = child.recv()
            except EOFError:  # for the case when the pipe has been closed
                child.close()
                break
            try:
                if cmd == 'getattr':
                    ret = getattr(env, args[0])
                elif cmd in method_name_list:
                    if cmd == 'set_seed':
                        ret = set_seed_fn(*args)
                    elif cmd == 'launch':
                        ret = launch_fn(*args)
                    elif cmd == 'reset':
                        ret = reset_fn(*args, **kwargs)
                    elif cmd == 'estimate':
                        ret = estimate_fn(*args, **kwargs)
                    elif cmd == 'update':
                        ret = update_fn(*args, **kwargs)                                                                                                        
                    elif args is None and kwargs is None:
                        ret = getattr(env, cmd)()
                    else:
                        ret = getattr(env, cmd)(*args, **kwargs)
                else:
                    raise KeyError("not support env cmd: {}".format(cmd))
                child.send(ret)
            except BaseException as e:
                # logging.debug("Sub env '{}' error when executing {}".format(str(env), cmd))
                # when there are some errors in env, worker_fn will send the errors to env manager
                # directly send error to another process will lose the stack trace, so we create a new Exception
                # logging.warning("subprocess exception traceback: \n" + traceback.format_exc())
                child.send(
                    e.__class__('\nEnv Process Exception:\n' + ''.join(traceback.format_tb(e.__traceback__)) + repr(e))
                )
            if cmd == 'close':
                child.close()
                break

    # override
    def close(self) -> None:
        """
        Overview:
            CLose the env manager and release all related resources.
        """
        self._closed = True
        for _, p in self._pipe_parents.items():
            p.send(['close', None, None])
        for env_id, p in self._pipe_parents.items():
            if not p.poll(5):
                continue
            tmp = p.recv()
        # # disable process join for avoiding hang
        # for p in self._subprocesses:
        #     p.join()
        for _, p in self._subprocesses.items():
            p.terminate()
        for _, p in self._pipe_parents.items():
            p.close()
            
    def launch(self):
        data_cloud = deepcopy(self.estimator.get_data())
        self.env_fns = [get_env_fn(self.cfg, self.estimator, i) for i in range(self.num_envs)]
        self._create_state()
        return data_cloud
        
    def poll(self, env_id):
        return self._pipe_parents[env_id].poll()
    
    def receive(self, env_id):
        return self._pipe_parents[env_id].recv()

    def estimate(self, env_id, actions):
        self._pipe_parents[env_id].send(['estimate', [actions], {}])

    def reset(self):
        for env_id in range(self.num_envs):
            self._pipe_parents[env_id].send(['reset', [], {}])
        self.receive_all()     

    def update(self, record):
        supporters, sum_errors, num_points = record.get_base()
        for env_id in range(self.num_envs):
            self._pipe_parents[env_id].send(['update', [supporters, sum_errors, num_points], {}])
        self.receive_all()

    # def update(self, estimator, instance_index):
    #     model = deepcopy(estimator.model)
    #     labels = deepcopy(estimator.labels)
    #     sum_errors = deepcopy(estimator.sum_errors)
    #     nearest_points = deepcopy(estimator.nearest_points)
    #     for env_id in range(self.num_envs):
    #         self._pipe_parents[env_id].send(['update', [model, sum_errors, nearest_points, labels, instance_index], {}])
    #     self.receive_all()

    def receive_all(self):
        for env_id in range(self.num_envs):
            try:
                self._pipe_parents[env_id].recv()
            except pickle.UnpicklingError as e:
                assert False                           
        
            
