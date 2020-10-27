"""
TODO: adapt for changing action space problems. Basically passing the set of actions, right now actions are assumed to
  be all integers between 0 and (number of actions - 1)
"""

from abc import ABC, abstractmethod

import numpy as np

from policies.utils import arbitrary_argmax


class ContextFreePolicy(ABC):
    """The abstract base class for policies that do not take the context (i.e. state) of actions into account, i.e. for
    context-free bandit algorithms."""

    @abstractmethod
    def action(self):
        pass

    @abstractmethod
    def update(self, a_t, r_t):
        """
        a_t = action taken at time step t.

        r_t = reward received (at time step t) after having taken action t.
        """
        pass


class RandomPolicy(ContextFreePolicy):
    """A policy that uniformly chooses an action at random and does not learn anything."""

    def __init__(self, num_actions):
        self.num_actions = num_actions

    def action(self):
        """It uniformly chooses an action at random."""
        return np.random.randint(0, self.num_actions)

    def update(self, a_t, r_t):
        """This policy does not learn, so it does not need to update any parameters."""
        pass


class GreedyPolicy(ContextFreePolicy):
    """A policy that chooses the action with the highest action value and then updates the estimate of the action value
    function using a moving average with a learning rate that is 1/n, where n is the number of times the action we are
    updating has been selected."""

    def __init__(self, num_actions):
        # Estimates of the action value function for every action, i.e. q(a), for all a.
        self.q = np.zeros(num_actions)

        # Keep track of the number of times each action has been taken.
        self.actions_count = np.zeros(num_actions, dtype=np.int)

    def action(self):
        """Choose and return the greedy action, i.e. the action with the current highest action value."""
        a_t = np.argmax(self.q)
        self.actions_count[a_t] += 1
        return a_t

    def update(self, a_t, r_t):
        """Updates the action value for action a_t, given reward r_t using a moving average as in equation 2.3 of
        "Reinforcement Learning: An Introduction" (2nd edition) by Sutton and Barto, i.e. the learning rate is 1/n,
        where n is the number of times action a_t has been taken so far, so this update is not appropriate for
        non-stationary problems. See section 2.5 of the same book for the non-stationary case."""
        n = self.actions_count[a_t]

        # self.q[a_t] = 1 / n * (self.q[a_t] * (n - 1) + r_t)
        # The previous statement is equivalent to (see equation 2.3 of the RL book).
        self.q[a_t] = self.q[a_t] + 1 / n * (r_t - self.q[a_t])


class EpsilonGreedyPolicy(ContextFreePolicy):
    """A policy that chooses an action in an epsilon-greedy fashion and updates the estimate of the action value
    function with a moving average (and constant learning rate that is annealed every time an action is taken)."""

    def __init__(self, num_actions, lr=0.1, epsilon=0.5, epsilon_annealing_factor=0.01):
        self.num_actions = num_actions
        self.q = np.zeros(num_actions)
        self.actions_count = np.zeros(num_actions, dtype=np.int)
        self.lr = lr  # learning rate (α)
        self.epsilon = epsilon  # the exploration rate (ϵ)
        self.epsilon_annealing_factor = epsilon_annealing_factor  # the annealing rate of the exploration rate
        self.t = 0  # the current time step (needed to anneal the exploration rate).

    def action(self):
        """Choose and return an action in epsilon-greedy fashion, i.e. with probability epsilon choose a random action,
        and with probability 1 - epsilon choose the greedy action (where ties, i.e. when there is more than one greedy
        action, are broken arbitrarily, e.g. uniformly)."""
        if np.random.uniform() < self.epsilon:  # With probability epsilon, uniformly choose an action at random.
            # Uniformly choose an action at random.
            a_t = np.random.randint(0, self.num_actions)
        else:
            # Choose one of the greedy actions (at random).
            a_t = arbitrary_argmax(self.q)

        # Increment the number of times action a_t has been selected.
        self.actions_count[a_t] += 1

        # Anneal the exploration rate.
        self.epsilon *= (1 - self.epsilon_annealing_factor) ** self.t

        # Increment the time step, which is needed to anneal the exploration rate (the next time an action is selected).
        self.t += 1

        return a_t

    def update(self, a_t, r_t):
        """Moving average update of the estimate of the action value function with a constant learning rate, which is
        also suited for non-stationary problems.

        See equation 2.5 (p. 32) of "Reinforcement Learning: An Introduction" (2nd edition) by Sutton and Barto."""
        self.q[a_t] = self.q[a_t] + self.lr * (r_t - self.q[a_t])


class UCBPolicy(ContextFreePolicy):

    def __init__(self, num_actions, lr=0.1, c=np.sqrt(2)):
        self.num_actions = num_actions
        self.q = np.zeros(num_actions)
        self.actions_count = np.zeros(num_actions, dtype=np.int)
        self.lr = lr
        self.t = 1

        # c in equation 2.10 of the RL book (2nd edition). In the original paper that introduced UCB1, i.e.
        # "Finite-time Analysis of the Multiarmed Bandit Problem"
        # https://homes.di.unimi.it/cesa-bianchi/Pubblicazioni/ml-02.pdf), c = sqrt(2).
        # See also https://ai.stackexchange.com/q/24221/2444
        self.c = c

    def action(self):
        """Select the action that maximizes the upper confidence bound and estimate of the action value function.

        See section 2.7 of "Reinforcement Learning: An Introduction" (2nd edition) by Sutton and Barto."""
        # ucb = upper confidence bound (for all actions at the current time step).
        # This can be interpreted as a measure of uncertainty (or variance) in the estimate of a's value.
        ucb = np.zeros(self.num_actions)

        for a in range(self.num_actions):
            # The higher the n (i.e. the number of times action a is selected), the smaller the associated ucb.
            n = self.actions_count[a]

            if n == 0:
                ucb[a] = np.inf
            else:
                # The higher the t (i.e. time step), given a fixed n, the higher the ucb.
                # The use of np.log (natural algorithm) means that increases (when t increases) get smaller over time,
                # but the increases can be unbounded.
                # All actions are eventually selected, but actions with lower value or that have already be selected
                # frequently will be selected with decreasing frequency over time.
                ucb[a] = np.sqrt(np.log(self.t) / n)

        a_t = np.argmax(self.q + (np.sqrt(self.c) * ucb))

        self.actions_count[a_t] += 1
        self.t += 1

        return a_t

    def update(self, a_t, r_t):
        """Updates the estimate of the action value function using a moving average."""
        # TODO: both this policy and EpsilonGreedyPolicy update the action value function in the same way, maybe we
        #  can abstract. Moreover, GreedyPolicy also uses the same update: the only thing that changes is that the
        #  learning rate is also moving (i.e. not constant).
        self.q[a_t] = self.q[a_t] + self.lr * (r_t - self.q[a_t])
