import numpy as np


def arbitrary_argmax(q):
    """Uniformly choose at random one of the indices of q where q is highest, if there is more than one index,
    otherwise just returns the index of the highest value.

    q = a 1-dimensional array.
    """
    all_greedy_actions = np.argwhere(q == np.amax(q)).flatten()
    a_t = np.random.choice(all_greedy_actions)
    return a_t
