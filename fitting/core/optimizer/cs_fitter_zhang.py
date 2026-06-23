import numpy as np
import setproctitle
from tools.tool import set_seed, get_seeds
from .record import Record
import pickle
from core.assessor import Assessor
import random


# optimization by the grey wolf optimizer
def grey_wolf_optimize(fitter):
    solution_dim = fitter.assessor.get_action_dim()
    lb = np.full(solution_dim, -1.)
    ub = np.full(solution_dim, 1.)
    # initialize alpha, beta, and delta_pos
    alpha_pos = np.zeros(solution_dim, dtype=np.float64)
    alpha_score = -float("inf")

    beta_pos = np.zeros(solution_dim, dtype=np.float64)
    beta_score = -float("inf")

    delta_pos = np.zeros(solution_dim, dtype=np.float64)
    delta_score = -float("inf")

    # # stagnation detection config
    # patience = fitter.cfg['fitter'].get('stagnation_patience', 5)
    # threshold = fitter.cfg['fitter'].get('stagnation_threshold', 0.0)

    # Initialize the positions of wolves
    num_wolves = fitter.population_size
    positions = np.zeros((num_wolves, solution_dim), dtype=np.float64)
    for i in range(solution_dim):
        positions[:, i] = (np.random.uniform(
            0, 1, num_wolves) * (ub[i] - lb[i]) + lb[i])

    max_estimation = fitter.cfg['fitter']['num_estimations']
    period = int(min(fitter.cfg['fitter'].get('period', 50000), max_estimation))
    iteration = 0
    stagnation_counter = 0
    prev_alpha_score = -float("inf")
    # Main loop
    for _ in range(0, max_estimation, num_wolves):
        scores = fitter.assess(solutions=positions)

        for i_wolf in range(0, num_wolves):
            score = scores[i_wolf]
            # Update Alpha, Beta, and Delta
            if score >= alpha_score:
                delta_score = beta_score  # Update delta
                delta_pos = beta_pos.copy()
                beta_score = alpha_score  # Update beta
                beta_pos = alpha_pos.copy()
                alpha_score = score  # Update alpha
                alpha_pos = positions[i_wolf, :].copy()
            elif score >= beta_score:
                delta_score = beta_score  # Update delta
                delta_pos = beta_pos.copy()
                beta_score = score  # Update beta
                beta_pos = positions[i_wolf, :].copy()
            elif score >= delta_score:
                delta_score = score  # Update delta
                delta_pos = positions[i_wolf, :].copy()
            else:
                pass
        iteration = (iteration + 1) % period

        # # stagnation detection and periodic restart
        # improvement = alpha_score - prev_alpha_score
        # prev_alpha_score = alpha_score
        # if improvement <= threshold:
        #     stagnation_counter += 1
        # else:
        #     stagnation_counter = 0
        # if stagnation_counter >= patience:
        #     iteration = 0
        #     stagnation_counter = 0

        # a decreases linearly from 2 to 0
        a = 2 - iteration * (2 / period)

        # Update the Position of wolves including omegas
        for i_wolf in range(0, num_wolves):
            for j in range(0, solution_dim):
                r1 = random.random()  # r1 is a random number in [0,1]
                r2 = random.random()  # r2 is a random number in [0,1]
                A1 = 2 * a * r1 - a  # Equation (3.3)
                C1 = 2 * r2  # Equation (3.4)
                # Equation (3.5)-part 1
                D_alpha = abs(C1 * alpha_pos[j] - positions[i_wolf, j])
                X1 = alpha_pos[j] - A1 * D_alpha  # Equation (3.6)-part 1
                r1 = random.random()
                r2 = random.random()
                A2 = 2 * a * r1 - a  # Equation (3.3)
                C2 = 2 * r2  # Equation (3.4)
                # Equation (3.5)-part 2
                D_beta = abs(C2 * beta_pos[j] - positions[i_wolf, j])
                X2 = beta_pos[j] - A2 * D_beta  # Equation (3.6)-part 2
                r1 = random.random()
                r2 = random.random()
                A3 = 2 * a * r1 - a  # Equation (3.3)
                C3 = 2 * r2  # Equation (3.4)
                # Equation (3.5)-part 3
                D_delta = abs(C3 * delta_pos[j] - positions[i_wolf, j])
                X3 = delta_pos[j] - A3 * D_delta  # Equation (3.5)-part 3

                positions[i_wolf, j] = (X1 + X2 + X3) / 3  # Equation (3.7)
                # Return back the wolves that go beyond the boundaries of the search space
                positions[i_wolf, j] = np.clip(
                    positions[i_wolf, j], lb[j], ub[j])


class Fitter:  # fitting by metaheuristic algorithms

    def __init__(self, cfg):
        self.cfg = cfg
        self.num_subpopulations = int(cfg['fitter']['num_subpopulations'])
        seeds = self.cfg.get('seeds', None)
        if seeds is None:
            seeds = get_seeds(self.num_subpopulations+1)
            self.cfg['seeds'] = seeds
            self.cfg['raw_seeds'] = None
        set_seed(seeds[-1])

        # set the title of the process
        setproctitle.setproctitle(
            cfg['estimator']['data_file'] + '-' + cfg['fitter']['optimizer'])

        self.population_size = cfg['fitter']['subpopulation_size'] * \
            self.num_subpopulations
        self.assessor = Assessor(cfg, num_evaluators=self.num_subpopulations)
        data_cloud = self.assessor.launch()
        self.record = Record(
            cfg, dimension=data_cloud.shape[1], estimator=self.assessor.estimator)
        self.record.data_cloud = data_cloud
        self.record.lower_bound = self.assessor.estimator.rule.lower_bound
        self.record.upper_bound = self.assessor.estimator.rule.upper_bound

        self.lb = None
        self.ub = None

        if self.cfg['fitter']['optimizer'] == 'cs':  # cuckoo search
            from .cs.optimizer import optimize
            self.optimize = optimize
        elif self.cfg['fitter']['optimizer'] == 'gwo':  # grey wolf optimizer
            self.optimize = grey_wolf_optimize
        else:
            assert False

    def assess(self, solutions):
        assert solutions.shape[0] == self.population_size
        subpopulation = self.cfg['fitter']['subpopulation_size']
        scores = np.zeros(self.population_size)
        for evaluator_id in range(self.num_subpopulations):
            part_solutions = solutions[evaluator_id *
                                       subpopulation:(evaluator_id+1)*subpopulation]
            self.assessor.evaluate(
                evaluator_id=evaluator_id, solutions=part_solutions)
        for evaluator_id in range(self.num_subpopulations):
            result = self.assessor.receive(evaluator_id)
            if isinstance(result, BaseException):
                raise result
            try:
                scores[evaluator_id*subpopulation:(
                    evaluator_id+1)*subpopulation], record = result
                self.record.update(record, subpopulation)
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"assessor.receive returned unexpected type {type(result).__name__}: {repr(result)}"
                ) from e
        return scores

    def fit(self):
        for i in range(self.cfg['fitter']['num_rounds']):
            self.record.round = i
            for k in range(self.cfg['fitter']['num_instances']):
                print(f'round {i} fitting for the model instance {k} begins')
                self.record.token_index = k
                self.assessor.update(self.record)
                self.record.estimation = 0
                # easydict 需要用string 作为key
                self.record.evolutions[f'round_{i}_instance_{k}'] = []

                self.optimize(self)
                print(f'round {i} fitting for the model instance {k} finished\n\n')

        print('the fitting is finished')
        return self.record.best_score

    def close(self):
        self.assessor.close()
        self.record.close()
