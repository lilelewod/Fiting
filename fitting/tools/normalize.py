def normalize(old, new_lb, new_ub):
    old_lb, old_ub = -1., 1.
    normalized = (new_ub - new_lb) * (old - old_lb) / (old_ub - old_lb) + new_lb
    return normalized


def normalize0(old, new_lb, new_ub):
    old_lb, old_ub = 0., 1.
    normalized = (new_ub - new_lb) * (old - old_lb) / (old_ub - old_lb) + new_lb
    return normalized

def normalize_oldb(old, new_lb, new_ub, old_lb, old_ub):
    normalized = (new_ub - new_lb) * (old - old_lb) / (old_ub - old_lb) + new_lb
    return normalized
