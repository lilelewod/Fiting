import numpy as np
from copy import deepcopy
from .record import SubRecord
from tools.tool import set_seed

class Environment:
    def __init__(self, cfg, estimator, env_id=0):
        self.cfg = deepcopy(cfg)
        self.env_id = deepcopy(env_id)
        set_seed(self.cfg['seeds'][self.env_id])
        self.record = SubRecord(cfg, env_id)
        self.records = []

        self.estimator = deepcopy(estimator)

    def reset(self):
        self.estimator.reset()

    def estimate(self, actions):  # interact with the environment for a number of episodes
        num_actions = actions.shape[0]
        scores = np.zeros((num_actions,))
        betters = np.full((num_actions,), False)
        self.record.episode = 0
        for i_action in range(num_actions):
            scores[i_action], betters[i_action] = self.react(
                actions[i_action, :])
        record = deepcopy(self.record) if np.any(betters) else None
        return scores, record

    def react(self, action):

        # caution: should normalize the action range to [-1, 1]
        # action = np.tanh(action_n[agent])
        assert not np.isnan(action).any()
        assert action.max() <= 1 and action.min() >= -1

        trait = self.estimator.parse(action=action)
        self.reset()
        self.estimator.current_dividing_level = -1
        self.estimator.generate()
        score = self.estimator.score
        better = self.record.update(score, self, trait=trait, action=action)
        return score, better

    def close(self):
        pass

    def update(self, model, sum_errors, nearest_points, labels, instance_index):
        self.estimator.update(model, sum_errors, nearest_points, labels, instance_index)
        self.records.append(self.record)
        self.record = SubRecord(self.cfg, self.env_id)
