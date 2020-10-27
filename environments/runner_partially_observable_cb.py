"""
Runner for portially observable reward CB problems.

Currently only supports shared contetxtual policies.
"""
import os

import numpy as np
import pandas as pd
import torch

from datautils.bandit_data import BanditData
from datautils.news.sample_data import sample_user_event
from environments.utils import create_if_not_exists
from policies.context_free_policies import (
    RandomPolicy
)
from policies.shared_contextual_policy import (FeedForwardNetwork,
                                               SharedLinUCBPolicy,
                                               SharedLinearGaussianThompsonSamplingPolicy,
                                               SharedNeuralPolicy)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
results_dir = os.path.abspath(os.path.join(root_dir, "results"))
create_if_not_exists(results_dir)


def simulate_partially_observable_cb(data_generator, n_samples, policies):
    """Simulator for POR CB problems.

    Runs for n_samples steps.
    """
    results = [None] * len(policies)
    for i, policy in enumerate(policies):
        results[i] = {}
        results[i]["reward"] = []

        t = 0
        t_1 = 0
        for uv in data_generator:
            # s_t = a list of article features
            u_t, S_t, r_acts, act_hidden = uv

            x_t = (u_t, S_t)
            a_t = policy.action(x_t)
            if a_t == act_hidden:
                assert act_hidden == r_acts[0]
                # useful
                r_t = r_acts[1]
                policy.update(a_t, x_t, r_t)
                results[i]["reward"].append(r_t)
                t_1 += 1
            else:
                # not useful
                # for off-policy learning
                pass

            t += 1

            if t_1 > n_samples:
                print("")
                print("{:.2f}% data useful".format(t_1 / t * 100))
                break

        results[i]["policy"] = policy
        rewards = results[i]["reward"]
        results[i]["cum_reward"] = np.cumsum(rewards)
        results[i]["CTR"] = np.array(rewards)

    return results


def run_partially_observable_cb(args):
    """Run partially observable reward problem."""
    n_rounds = args.n_rounds
    user_event_generator = sample_user_event()

    n_actions = 20
    context_dim = 6 + 6

    rp = RandomPolicy(n_actions)

    linucbp = SharedLinUCBPolicy(
        context_dimension=context_dim,
        delta=0.25,
        updating_starts_at=args.train_starts_at,
        update_frequency=args.train_freq
    )

    lgtsp = SharedLinearGaussianThompsonSamplingPolicy(
        context_dim=context_dim,
        eta_prior=6.0,
        lambda_prior=0.25,
        train_starts_at=args.train_starts_at,
        posterior_update_freq=args.train_freq
    )

    # prepore neural policy

    np.random.seed(0)
    torch.manual_seed(0)

    batch_size = args.batch_size
    set_gpu = args.cuda
    eta = args.eta
    gamma = args.gamma

    grad_clip = args.grad_clip
    grad_clip_norm = args.grad_clip_norm
    grad_clip_value = args.grad_clip_value

    grad_noise = args.grad_noise

    ffn = FeedForwardNetwork(input_dim=context_dim,
                             hidden_dim=64,
                             output_dim=1,
                             n_layer=3,
                             learning_rate=args.lr,
                             set_gpu=set_gpu,
                             grad_noise=grad_noise,
                             gamma=gamma,
                             eta=eta,
                             grad_clip=grad_clip,
                             grad_clip_norm=grad_clip_norm,
                             grad_clip_value=grad_clip_value,
                             weight_decay=args.weight_decay,
                             debug=args.debug)

    # batch data loader
    bandit_data = BanditData(batch_size, epoch_len=16)
    # 16 x 64

    neuralp = SharedNeuralPolicy(ffn,
                                 bandit_data,
                                 train_starts_at=args.train_starts_at,
                                 train_freq=args.train_freq,
                                 set_gpu=set_gpu)

    policies = [rp, linucbp, lgtsp, neuralp]
    policy_names = ["rp", "linucbp", "lgtsp", "neuralp"]

    results = simulate_partially_observable_cb(user_event_generator, n_rounds, policies)

    return results, policies, policy_names


def write_results_por_cb(results, policies, policy_names, trial_idx, args):
    """Write results to csv for pab_cb."""
    # log results
    cumulative_reward = None
    for i in range(len(policies)):
        cr = results[i]["cum_reward"][:, None]
        if cumulative_reward is None:
            cumulative_reward = cr
        else:
            cumulative_reward = np.hstack((cumulative_reward, cr))

    df = pd.DataFrame(cumulative_reward, columns=policy_names)
    df.to_csv("{}/{}.cumulative_reward.{}.csv".format(results_dir, args.task, trial_idx), header=True, index=False)

    CTR_data = None  # Click through rate
    for i in range(len(policies)):
        cr = results[i]["CTR"][:, None]
        if CTR_data is None:
            CTR_data = cr
        else:
            CTR_data = np.hstack((CTR_data, cr))

    df = pd.DataFrame(CTR_data, columns=policy_names)
    df.to_csv("{}/{}.CTR.{}.csv".format(results_dir, args.task, trial_idx), header=True, index=False)
