import numpy as np
from .gd_initializer import gradient_initialize_actions
from copy import deepcopy
import math


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


def initialize(fitter, n):
    fitter.last_nest = [None, ] * fitter.cfg['fitter']['num_instances']
    fitness = np.full(n, -np.inf)        
    if fitter.last_nest[fitter.record.token_index] is None:
        dim = fitter.assessor.get_action_dim()
        fitter.lb = np.full(dim, -1.)
        fitter.ub = np.full(dim, 1.)                   
        nest = np.zeros((n, dim))
        for i in range(n):
            nest[i, :] = fitter.lb + (fitter.ub-fitter.lb) * np.random.random_sample(fitter.lb.shape)
        gd_actions = gradient_initialize_actions(
            fitter.assessor.estimator,
            min(n, int(fitter.cfg.get('gradient_initializer', {}).get('num_seeds', 0))),
            fitter.cfg,
            rng=np.random.default_rng(fitter.cfg['seeds'][-1] + 1009 + fitter.record.token_index),
        )
        if gd_actions.shape[0] > 0:
            nest[:gd_actions.shape[0], :] = gd_actions
    else:
        nest = fitter.last_nest[fitter.record.token_index]
    best_score, best_nest, nest, fitness = get_best_nest(fitter, nest, nest, fitness)
    return best_score, best_nest, nest, fitness


def get_best_nest(fitter, nest, new_nest, fitness):
    scores = fitter.assess(solutions=new_nest)
    mask = scores >= fitness
    fitness[mask] = scores[mask]
    nest[mask, :] = new_nest[mask, :]
    max_index = np.argmax(fitness)
    f_max = fitness[max_index]
    best = nest[max_index, :]
    return f_max, best, nest, fitness    


def optimize(fitter):

    n = fitter.population_size
    pa = 0.25

    best_score, best_nest, nest, fitness = initialize(fitter, n)
    max_estimations = fitter.cfg['fitter']['num_estimations']
    estimations = 0
    best_score = 0.
    while estimations < max_estimations:
        new_nest = get_cuckoos(nest, best_nest, fitter.lb, fitter.ub)
        f_new, best_nest, nest, fitness = get_best_nest(fitter, nest, new_nest, fitness)    
        new_nest = empty_nests(nest, fitter.lb, fitter.ub, pa)
        f_new, best_nest, nest, fitness = get_best_nest(fitter, nest, new_nest, fitness)
        estimations += 2 * n
    fitter.last_nest[fitter.record.token_index] = nest

        
        
