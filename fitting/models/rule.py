import numpy as np
from copy import copy


class Token:  # i.e., model instance

    def __init__(self, dimension):
        self.points = np.empty((0, dimension), dtype=np.float32)
        self.supporters = None
        self.sum_errors = None
        self.trait = None
        self.measure = None
        self.color = np.empty(0, dtype=np.int64)
        self.image = None
        self.action = None


# Non-terminal rule base
class Rule:
    name = 'rule'

    def __init__(self, estimator=None):
        self.estimator = estimator
        self.is_terminal = False

    def generate(self, **kwargs):
        assert False


# Terminal rule base of geometric model
class ModelRule(Rule):
    def __init__(self, estimator=None):
        Rule.__init__(self, estimator)
        self.is_terminal = True
        self.top_level = None
        self.top_dividing_level_all_reached = True
        self.current_dividing_level = -1

    def compute_current_dividing_level(self):
        level = copy(self.top_level)
        suggest_level = self.estimator.current_dividing_level
        if suggest_level < 0 or np.all(suggest_level >= self.top_level):
            self.top_dividing_level_all_reached = True
        else:
            level[suggest_level < self.top_level] = suggest_level
            self.top_dividing_level_all_reached = False
        self.current_dividing_level = level       
        return level
