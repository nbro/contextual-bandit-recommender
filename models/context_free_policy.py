"""
"""
import numpy as np

# context-free

# no exploration
class RandomPolicy(object):
    def __init__(self, n_actions):
        self._n_actions = n_actions

    def choose_action(self, c_t):
        return np.random.choice(np.arange(self._n_actions))

    def update(self, a_t, c_t, r_t):
        pass

# no exploration
class SampleMeanPolicy(object):
    def __init__(self, n_actions, lr=0.1):
        self._Q = np.zeros(n_actions)
        self._act_count = np.zeros(n_actions)
        self._lr = lr

    def choose_action(self, c_t):
        a_t = np.argmax(self._Q)
        self._act_count[a_t] += 1
        return a_t

    def update(self, a_t, c_t, r_t):
        # ignores context
        n_j = self._act_count[a_t]
        self._Q[a_t] = 1/n_j * (self._Q[a_t] *(n_j - 1) + r_t)


# exploration (with annealing)
class EpsilonGreedyPolicy(object):
    def __init__(self, n_actions, lr=0.1, epsilon=0.5, eps_anneal_factor=0.01):
        self._n_actions = n_actions
        self._Q = np.zeros(n_actions)
        self._act_count = np.zeros(n_actions)
        self._lr = lr
        self._eps = epsilon
        self._t = 0
        self._eps_anneal_factor = eps_anneal_factor

    def choose_action(self, c_t):
        u = np.random.uniform()
        if u > self._eps:
            a_t = np.argmax(self._Q)
        else:
            # choose random
            a_t = np.random.choice(np.arange(self._n_actions))
        self._act_count[a_t] += 1

        # anneal eps
        self._eps *= (1 - self._eps_anneal_factor)**self._t

        self._t += 1
        return a_t

    def update(self, a_t, c_t, r_t):
        # ignores context
        n_j = self._act_count[a_t]
        # standard approach
        # self._Q[a_t] = 1/n_j * (self._Q[a_t] *(n_j - 1) + r_t)
        # robust for non-stationary
        self._Q[a_t] = self._Q[a_t] + self._lr * (r_t - self._Q[a_t])


# exploration with ucb
class UCBPolicy(object):
    def __init__(self, n_actions, lr=0.1):
        self._n_actions = n_actions
        self._Q = np.zeros(n_actions)
        self._act_count = np.zeros(n_actions, dtype=int)
        self._lr = lr
        self._t = 1

    def choose_action(self, c_t):
        ubc_t = np.zeros(self._n_actions)
        for a_j in range(self._n_actions):
            # compute upper bound
            n_j = self._act_count[a_j]
            if n_j == 0:
                ubc_t[a_j] = np.inf
            else:
                ubc_t[a_j] = np.sqrt(2*np.log(self._t)/n_j)


        #if self._t % 10000 == 0:
        #    print("ubc {} at {}".format(ubc_t, self._t))
        #    print("act count", self._act_count)
        a_t = np.argmax(self._Q + ubc_t)
        self._act_count[a_t] += 1
        self._t += 1
        return a_t

    def update(self, a_t, c_t, r_t):
        # ignores context
        n_j = self._act_count[a_t]
        # standard approach
        # self._Q[a_t] = 1/n_j * (self._Q[a_t] *(n_j - 1) + r_t)
        # robust for non-stationary
        self._Q[a_t] = self._Q[a_t] + self._lr * (r_t - self._Q[a_t])


