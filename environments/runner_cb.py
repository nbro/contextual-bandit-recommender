"""
Runner for fully observable reward CB problems.
"""
import os

import numpy as np
import pandas as pd

from datautils.mushroom.sample_data import sample_mushroom
from datautils.preprocessing import load_data
from datautils.synthetic.sample_data import sample_synthetic
from environments.utils import create_if_not_exists
from policies.context_free_policies import (
    EpsilonGreedyPolicy,
    UCBPolicy,
    ContextFreePolicy
)
from policies.disjoint_contextual_policy import (
    LinUCBPolicy,
    LinearGaussianThompsonSamplingPolicy,
)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
results_dir = os.path.abspath(os.path.join(root_dir, "results"))

create_if_not_exists(results_dir)


def simulate_contextual_bandit(data, n_samples, policies):
    """Simulator for for contextual bandit (CB) problems.

    Runs n_samples steps.
    """
    results = [None] * len(policies)

    for i, policy in enumerate(policies):

        # Create a dictionary for policy i where we save different statistics related to it (such as the regret).
        results[i] = {}

        # log contains a_t, optimal_a_t, r_t, regret_t
        results[i]["log"] = np.zeros((4, n_samples))

        t = 0

        for x_t, actions_to_reward, optimal_a_t, _ in zip(*data):
            if isinstance(policy, ContextFreePolicy):
                a_t = policy.action()
            else:
                a_t = policy.action(x_t)  # x_t is the context at time step t.

            r_t = actions_to_reward[a_t]  # reward for each of the actions.

            if isinstance(policy, ContextFreePolicy):
                policy.update(a_t, r_t)
            else:
                policy.update(a_t, x_t, r_t)

            # Get the reward for the optimal action.
            r_t_opt = actions_to_reward[optimal_a_t]  # optimal_a_t optimal action at time step t.

            # Compute the regret as the difference between the optimal reward and the reward for taking the action
            # according to the given behaviour policy.
            regret_t = r_t_opt - r_t

            # Save the results for policy i.
            results[i]["log"][:, t] = [a_t, optimal_a_t, r_t, regret_t]

            t += 1

        results[i]["policy"] = policy

        # All regrets for all time steps
        regrets = results[i]["log"][3, :]

        # https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html
        # TODO: Why are we interested in the cumulative regret and why do we compute it like that?
        # TODO: for example, how does this relate to equation 1 of the paper "A Contextual-Bandit Approach to
        #  Personalized News Article Recommendation"
        results[i]["cum_regret"] = np.cumsum(regrets)

        # results[i]["simple_regret"] = np.sum(regrets[-500:])

    return results


# This function is called from main.py.
def run_cb(args):
    """Run fully observable reward CB problems."""
    task = args.task
    n_rounds = args.n_rounds

    # https://archive.ics.uci.edu/ml/datasets/mushroom
    if task == "mushroom":
        # X.shape = (8123, 117)
        X, y = load_data(name="mushroom")

        # Each observation/feature vector is an array of 117 elements.
        # Although the mushrooms dataset only contains 22 input features, 117 is because we convert the initial vectors
        # to indicator variables.
        # See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html
        context_dim = 117

        n_actions = 2  # 2 actions: eat and not eat.

        samples = sample_mushroom(X,
                                  y,
                                  n_rounds,

                                  # Defines the different types of rewards.
                                  r_eat_good=10.0,
                                  r_eat_bad_lucky=10.0,
                                  r_eat_bad_unlucky=-50.0,
                                  r_eat_bad_lucky_prob=0.7,
                                  r_no_eat=0.0
                                  )

        # samples is a tuple
        # samples[0].shape = (600, 117) => 600 contexts, each of which of 117 dimensions (i.e. the feature vector).
        # samples[1].shape = (600, 2) => rewards for each of the 2 actions
        # samples[2].shape = (600,) => optimal action for each of the contexts
        # samples[3].shape = (600,)
        # 600 is the number of rounds (args.n_rounds)

    elif task == "synthetic":
        n_actions = 5
        context_dim = 10
        sigma = 1.0  # set low covariance
        samples = sample_synthetic(n_rounds, n_actions, context_dim, sigma)

    else:
        raise NotImplementedError("other tasks have not yet been implemented")

    # define a solver

    # Context-free bandit policies
    egp = EpsilonGreedyPolicy(n_actions,
                              lr=0.001,
                              epsilon=0.5,
                              epsilon_annealing_factor=0.001)

    ucbp = UCBPolicy(num_actions=n_actions,
                     lr=0.001)

    # Contextual bandit policies
    linucbp = LinUCBPolicy(num_actions=n_actions,
                           context_dimension=context_dim,
                           delta=0.001,
                           updating_starts_at=100,
                           update_frequency=5)

    lgtsp = LinearGaussianThompsonSamplingPolicy(n_actions=n_actions,
                                                 context_dim=context_dim,
                                                 eta_prior=6.0,
                                                 lambda_prior=0.25,
                                                 train_starts_at=100,
                                                 posterior_update_freq=5,
                                                 lr=0.05)

    policies = [egp, ucbp, linucbp, lgtsp]
    policy_names = ["egp", "ucbp", "linucbp", "lgtsp"]

    # simulate a bandit over n_rounds steps
    results = simulate_contextual_bandit(samples, n_rounds, policies)

    # results contains a list of dictionaries, one for each policy. Each of these dictionaries contains statistics
    # associated with the results (e.g. regret for each time step) of running the corresponding policy with the given
    # data.
    return results, policies, policy_names


def write_results_cb(results, policies, policy_names, trial_idx, args):
    """Writes results to csv files."""
    # log results
    cumulative_regret_data = None

    actions_data = None

    for i in range(len(policies)):
        # Cumulative regret (where regret is true reward - reward).
        # None adds an extra dimension, this is done so that we can stack all the cumulative regrets as columns.
        cr = results[i]["cum_regret"][:, None]
        # print(cr.shape)

        if cumulative_regret_data is None:
            cumulative_regret_data = cr
        else:
            cumulative_regret_data = np.hstack((cumulative_regret_data, cr))

        # Save the actions taken by the policy i
        # 0 were the actions in the simulate_cb method above.
        acts = results[i]["log"][0, :][:, None]

        if actions_data is None:
            actions_data = acts
        else:
            actions_data = np.hstack((actions_data, acts))

    # select the optimal actions.

    acts_opt = results[0]["log"][1, :][:, None]

    # Actions taken by all policies and optimal actions.
    actions_data = np.hstack((actions_data, acts_opt))

    df = pd.DataFrame(cumulative_regret_data, columns=policy_names)
    df.to_csv("{}/{}.cumulative_regret.{}.csv".format(results_dir, args.task, trial_idx), header=True, index=False)

    df = pd.DataFrame(actions_data, columns=policy_names + ["opt_p"])
    df.to_csv("{}/{}.actions.{}.csv".format(results_dir, args.task, trial_idx), header=True, index=False)
