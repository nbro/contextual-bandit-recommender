"""
The policies that use user contexts.

The current implementations build disjoint policies for each action.

The current implementations do not accept item (action) contexts.

@todo: combine this with action_context_policies.
       1. disjoint, 2. shared.
"""
from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import invgamma

from policies.utils import arbitrary_argmax


class ContextualPolicy(ABC):
    """The abstract base class for policies that do take the context (i.e. state) of actions into account, i.e. for
    contextual bandit algorithms."""

    @abstractmethod
    def action(self, x_t):
        """x_t = the context at time step t."""
        pass

    @abstractmethod
    def update(self, a_t, x_t, r_t):
        """
        a_t = action taken at time step t.

        x_t = the context at time step t.

        r_t = reward received (at time step t) after having taken action t.
        """
        pass


class LinUCBPolicy(ContextualPolicy):
    """The implementation of algorithm 1 in the paper "A Contextual-Bandit Approach to Personalized News Article
    Recommendation" (2010, http://rob.schapire.net/papers/www10.pdf). So, in this case, we do not have a set of
    parameters that are shared across actions, but each action contains its separate vector of parameters.

    The parameters are only updated periodically, for computational reasons."""

    def __init__(self, num_actions, context_dimension, delta=0.2, updating_starts_at=500, update_frequency=50):
        self.num_actions = num_actions

        self.actions_count = np.zeros(num_actions, dtype=np.int)

        # +1 because of the bias.
        self.d = context_dimension + 1

        # Initially, for each action, we define a matrix A, i.e. A_a, as an identity with the same dimension as the
        # context. See algorithm 1 of "A Contextual-Bandit Approach to Personalized News Article Recommendation" (2010,
        # http://rob.schapire.net/papers/www10.pdf).
        self.A = [np.identity(self.d) for _ in range(self.num_actions)]

        self.A_inv = np.linalg.inv(self.A)

        # Response vector (e.g., in the context of item recommendation, this vector may contain whether an article was
        # clicked or not).
        self.b = [np.zeros(self.d) for _ in range(self.num_actions)]

        # Estimate the theta, i.e. the parameters that the define the expected reward given the context, for each
        # action a. Note that we have 1 theta for each action, i.e. different actions do NOT share the parameters.
        self.theta = [self.A_inv[action].dot(self.b[action]) for action in range(self.num_actions)]

        # alpha value below equation 4 of the paper "A Contextual-Bandit Approach to Personalized News Article
        # Recommendation" (2010, http://rob.schapire.net/papers/www10.pdf). In algorithm 1 of the same paper, alpha is
        # given as a hyper-parameter, while here delta is given as hyper-parameter, given that alpha is defined as a
        # function of delta. alpha is used to estimate the action value function at each time step. See equation 5 in
        # the same paper and the method self.action below.
        self.alpha = 1 + np.sqrt(np.log(2 / delta) / 2)

        self.t = 0

        # These parameters are used in self.update to decide when to periodically update the parameters.
        self.update_frequency = update_frequency  # Every how many time steps the self.theta should be updated.

        # TODO: why do we need this?
        self.updating_starts_at = updating_starts_at  # The time step to start updating self.theta.

    def action(self, x_t):
        """Select an action according to equation 5 of paper "A Contextual-Bandit Approach to Personalized News Article
        Recommendation" (2010, http://rob.schapire.net/papers/www10.pdf). See also algorithm 1 of the same paper.

        x_t = is the context at time step t."""

        # Lines 8 and 9 of algorithm 1.
        x_t = np.append(x_t, 1)  # For the bias.

        q = np.zeros(self.num_actions)
        ubc = np.zeros(self.num_actions)

        # Compute the estimate of action value function for each action using the UCB-based method described in section
        # 3 of the paper "A Contextual-Bandit Approach to Personalized News Article Recommendation"
        # (2010, http://rob.schapire.net/papers/www10.pdf).
        for action in range(self.num_actions):
            k = x_t.T.dot(self.A_inv[action]).dot(x_t)
            ubc[action] = self.alpha * np.sqrt(k)

            q[action] = self.theta[action].dot(x_t) + ubc[action]

        # Choose one of the greedy actions (at random) according equation 5 of the paper.
        # Line 11 of algorithm 1.
        a_t = arbitrary_argmax(q)

        self.t += 1

        return a_t

    def update(self, a_t, x_t, r_t):
        """Update self.theta, i.e. the parameters that define the estimate of the action value function given some
        context.

        a_t = action taken at time step t.

        x_t = the context at time step t.

        r_t = reward received (at time step t) after having taken action t."""
        x_t = np.append(x_t, 1)  # d x 1

        # Line 12 of algorithm 1.
        self.A[a_t] += x_t.dot(x_t.T)  # d x d

        # Line 13 of algorithm 1.
        self.b[a_t] += r_t * x_t  # d x 1

        if self.t < self.updating_starts_at:
            return

        if self.t % self.update_frequency == 0:
            # Line 8 of algorithm 1.
            # TODO: use lstsq (https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html) to handle
            #  over/under determined systems?
            self.A_inv[a_t] = np.linalg.inv(self.A[a_t])
            self.theta[a_t] = self.A_inv[a_t].dot(self.b[a_t])


class LinearGaussianThompsonSamplingPolicy(ContextualPolicy):
    """
    Linear Gaussian Thompson Sampling policy.

    A Bayesian approach for inferring the true reward distribution model.

    Implements a Bayesian linear regression with a conjugate prior [1].

    Model: Gaussian: R_t = W*x_t + eps, eps ~ N(0, sigma^2 I)
    Model Parameters: mu, cov
    Prior on the parameters
    p(w, sigma^2) = p(w|sigma^2) * p(sigma^2)

    1. p(sigma^2) ~ Inverse Gamma(a_0, b_0)
    2. p(w|sigma^2) ~ N(mu_0, sigma^2 * precision_0^-1)

    For computational reasons,

    1. model updates are done periodically.
    2. for large sample cases, batch_mode is enbabled.

    [1]: https://en.wikipedia.org/wiki/Bayesian_linear_regression#Conjugate_prior_distribution

    """

    def __init__(self,
                 n_actions,
                 context_dim,
                 eta_prior=6.0,
                 lambda_prior=0.25,
                 train_starts_at=500,
                 posterior_update_freq=50,
                 batch_mode=True,
                 batch_size=512,
                 lr=0.1):
        self._t = 1
        self._update_freq = posterior_update_freq
        self._train_starts_at = train_starts_at

        self._n_actions = n_actions
        # bias
        self._d = context_dim + 1

        # inverse gamma prior
        self._a_0 = eta_prior
        self._b_0 = eta_prior
        self._a_list = [eta_prior] * n_actions
        self._b_list = [eta_prior] * n_actions

        # conditional Gaussian prior
        self._sigma_sq_0 = invgamma.rvs(eta_prior, eta_prior)
        self._lambda_prior = lambda_prior
        # precision_0 shared for all actions
        self._precision_0 = self._sigma_sq_0 / self._lambda_prior * np.eye(self._d)

        # initialized at mu_0
        self._mu_list = [
            np.zeros(self._d)
            for _ in range(n_actions)
        ]

        # initialized at cov_0
        self._cov_list = [
            1.0 / self._lambda_prior * np.eye(self._d)
            for _ in range(n_actions)
        ]

        # remember training data
        self._train_data = [None] * n_actions

        # for computational efficiency
        # train on a random subset
        self._batch_mode = batch_mode
        self._batch_size = batch_size
        self._lr = lr

    def _update_posterior(self, act_t, X_t, r_t_list):
        cov_t = np.linalg.inv(np.dot(X_t.T, X_t) + self._precision_0)
        mu_t = np.dot(cov_t, np.dot(X_t.T, r_t_list))
        a_t = self._a_0 + self._t / 2

        # mu_0 simplifies some terms
        r = np.dot(r_t_list, r_t_list)
        precision_t = np.linalg.inv(cov_t)
        b_t = self._b_0 + 0.5 * (r - np.dot(mu_t.T, np.dot(precision_t, mu_t)))

        self._cov_list[act_t] = cov_t
        self._mu_list[act_t] = mu_t
        self._a_list[act_t] = a_t
        self._b_list[act_t] = b_t

        if self._batch_mode:
            # learn bit by bit
            self._cov_list[act_t] = cov_t * self._lr + self._cov_list[act_t] * (1 - self._lr)
            self._mu_list[act_t] = mu_t * self._lr + self._mu_list[act_t] * (1 - self._lr)
            self._a_list[act_t] = a_t * self._lr + self._a_list[act_t] * (1 - self._lr)
            self._b_list[act_t] = b_t * self._lr + self._b_list[act_t] * (1 - self._lr)
        else:
            self._cov_list[act_t] = cov_t
            self._mu_list[act_t] = mu_t
            self._a_list[act_t] = a_t
            self._b_list[act_t] = b_t

    def _sample_posterior_predictive(self, x_t, n_samples=1):
        # p(sigma^2)
        sigma_sq_t_list = [
            invgamma.rvs(self._a_list[j], scale=self._b_list[j])
            for j in range(self._n_actions)
        ]

        try:
            # p(w|sigma^2) = N(mu, sigam^2 * cov)
            W_t = [
                np.random.multivariate_normal(
                    self._mu_list[j], sigma_sq_t_list[j] * self._cov_list[j]
                )
                for j in range(self._n_actions)
            ]
        except np.linalg.LinAlgError as e:
            print("Error in {}".format(type(self).__name__))
            print('Errors: {}.'.format(e.args[0]))
            W_t = [
                np.random.multivariate_normal(
                    np.zeros(self._d), np.eye(self._d)
                )
                for i in range(self._n_actions)
            ]

        # p(r_new | params)
        mean_t_predictive = np.dot(W_t, x_t)
        cov_t_predictive = sigma_sq_t_list * np.eye(self._n_actions)
        r_t_estimates = np.random.multivariate_normal(
            mean_t_predictive,
            cov=cov_t_predictive, size=1
        )
        r_t_estimates = r_t_estimates.squeeze()

        assert r_t_estimates.shape[0] == self._n_actions

        return r_t_estimates

    def action(self, x_t):
        x_t = np.append(x_t, 1)
        r_t_estimates = self._sample_posterior_predictive(x_t)
        act = np.argmax(r_t_estimates)

        self._t += 1

        return act

    def update(self, a_t, x_t, r_t):
        self._set_train_data(a_t, x_t, r_t)
        # sample model parameters
        # p(w, sigma^2 | X_t, r_vec_t)
        X_t, r_t_list = self._get_train_data(a_t)
        n_samples = X_t.shape[0]

        if self._t < self._train_starts_at:
            return

        # posterior update periodically per action
        if n_samples % self._update_freq == 0:
            self._update_posterior(a_t, X_t, r_t_list)

    def _get_train_data(self, a_t):
        return self._train_data[a_t]

    def _set_train_data(self, a_t, x_t, r_t):
        # add bias
        x_t = np.append(x_t, 1)

        if self._train_data[a_t] is None:
            X_t = x_t[None, :]
            r_t_list = np.array([r_t])

        else:
            X_t, r_t_list = self._train_data[a_t]
            n = X_t.shape[0]
            X_t = np.vstack((X_t, x_t))
            assert X_t.shape[0] == (n + 1)
            assert X_t.shape[1] == self._d

            r_t_list = np.append(r_t_list, r_t)

        # train on a random batch
        n_samples = X_t.shape[0]
        if self._batch_mode and self._batch_size < n_samples:
            indices = np.arange(self._batch_size)
            batch_indices = np.random.choice(indices,
                                             size=self._batch_size,
                                             replace=False)
            X_t = X_t[batch_indices, :]
            r_t_list = r_t_list[batch_indices]

        self._train_data[a_t] = (X_t, r_t_list)
