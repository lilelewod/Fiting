import numpy as np
from copy import deepcopy
from .record import SubRecord
from tools.tool import set_seed

class Evaluator:  # evaluate a subpopulation
    def __init__(self, cfg, estimator, evaluator_id=0, faiss_lock=None):
        self.cfg = deepcopy(cfg)
        self.evaluator_id = deepcopy(evaluator_id)
        set_seed(self.cfg['seeds'][self.evaluator_id])
        self.record = SubRecord(cfg, evaluator_id)
        self.records = []
        self._faiss_lock = faiss_lock

        self.estimator = deepcopy(estimator)
        self.estimator._faiss_lock = self._faiss_lock
        self.estimator.prepare_model_to_data_backend()

    def reset(self):
        self.estimator.reset()

    def evaluate(self, solutions):  # evaluate a number of solutions
        num_solutions = solutions.shape[0]
        scores = np.zeros((num_solutions,))
        betters = np.full((num_solutions,), False)
        self.record.estimation = 0
        self.record.early_tested = 0
        self.record.early_rejected = 0
        self.record.surface_sampling_time = 0.0
        self.record.model_to_data_error_time = 0.0
        for i_solution in range(num_solutions):
            scores[i_solution], betters[i_solution] = self.estimate(
                solutions[i_solution, :])
        record = deepcopy(self.record)
        return scores, record

    def accumulate_timing(self):
        self.record.surface_sampling_time += getattr(
            self.estimator, 'last_surface_sampling_time', 0.0)
        self.record.model_to_data_error_time += getattr(
            self.estimator, 'last_model_to_data_error_time', 0.0)

    def should_early_reject(self):
        self.record.early_tested += 1
        mode = self.cfg['estimator'].get('early_rejection_mode', 'mm')
        alpha = self.cfg['estimator'].get('early_rejection_alpha', 0.5)
        beta = self.cfg['estimator'].get('early_rejection_beta', 1.5)
        error_scale = self.cfg['estimator'].get('early_rejection_error_scale', None)
        if error_scale is None:
            error_scale = self.cfg['estimator'].get('early_rejection_band', None)
        if error_scale is None:
            error_scale = self.estimator.data_resolution

        if mode == 'mm':
            return self.estimator.score_mm < self.record.best_score_mm
        if mode == 'npre_score':
            return (
                self.record.best_score > 0
                and self.estimator.score_npre < alpha * self.record.best_score
            )
        if mode == 'npre_error':
            return (
                self.record.best_single_model_error < float('inf')
                and self.estimator.single_model_error > beta * self.record.best_single_model_error
            )
        if mode in ('npre_error_prob_exp', 'npre_error_prob'):
            best_error = self.record.best_single_model_error + self.estimator.model_resolution
            current_error = self.estimator.single_model_error
            if best_error < float('inf') and current_error > best_error:
                denominator = max(float(error_scale), 1e-12)
                reject_probability = 1.0 - np.exp(-(current_error - best_error) / denominator)
                reject_probability = float(np.clip(reject_probability, 0.0, 1.0))
                return np.random.rand() < reject_probability
            return False
        if mode == 'npre_error_prob_linear':
            best_error = self.record.best_single_model_error
            current_error = self.estimator.single_model_error
            if best_error < float('inf') and current_error > best_error:
                beta_margin = max(beta - 1.0, 1e-12)
                denominator = beta_margin * max(abs(best_error), 1e-12)
                reject_probability = min(1.0, (current_error - best_error) / denominator)
                return np.random.rand() < reject_probability
            return False
        if mode == 'npre_hybrid':
            bad_score = (
                self.record.best_score > 0
                and self.estimator.score_npre < alpha * self.record.best_score
            )
            bad_error = (
                self.record.best_single_model_error < float('inf')
                and self.estimator.single_model_error > beta * self.record.best_single_model_error
            )
            return bad_score and bad_error
        raise ValueError(f'Unknown early_rejection_mode: {mode}')

    def estimate(self, solution):

        # caution: should normalize the action range to [-1, 1]
        # action = np.tanh(action_n[agent])
        assert not np.isnan(solution).any()
        assert solution.max() <= 1 and solution.min() >= -1

        self.estimator.parse(solution=solution)
        levels = int(self.cfg['estimator'].get('early_rejection_levels', 0))
        levels = max(levels, 0)
        for level in range(levels, -1, -1):
            self.reset()
            self.estimator.generate(sampling_level=level)
            self.accumulate_timing()            
            if level == 0:  # the finest level
                score = self.estimator.score
                better = self.record.update(score, self.estimator)
                return score, better
            elif self.should_early_reject():
                self.record.early_rejected += 1
                score = -1
                return score, False

    def close(self):
        pass

    def update(self, supporters, sum_errors, num_points):
        self.estimator.update(supporters, sum_errors, num_points)
        self.records.append(self.record)
        self.record = SubRecord(self.cfg, self.evaluator_id)
