# Code that attempts to reproduce section 2.3 from Reinforcement Learning: An Introduction (2nd edition)
# by Barto and Sutton.

import matplotlib.pyplot as plt
import numpy as np

# These are the hyper-parameters used by Sutton and Barto.

NUM_ACTIONS = 10  # k
NUM_TIME_STEPS = 1000
NUM_RUNS = 2000

# epsilon == 0.0 means that the algorithm always chooses the greedy action
EPSILONS = np.array([0.01, 0.1, 0.0])


def sample_average(num_time_steps=NUM_TIME_STEPS, num_runs=NUM_RUNS, epsilon=EPSILONS[2]):
    # Equations 2.1, 2.2

    def get_optimal_action(q):
        return np.argmax(q)

    def sample_action(estimated_q):
        if np.random.random() < epsilon:
            # Return a random action (i.e. a number between 0 and 9, both inclusive) with probability epsilon.
            return np.random.randint(0, NUM_ACTIONS)
        else:
            # Return the greedy action.
            first_greedy_action = np.argmax(estimated_q)

            # Find all actions that have the same estimated q value (because they are also greedy).
            # If there is another greedy action, randomly choose among the greedy actions.
            greedy_actions = np.where(estimated_q == first_greedy_action)[0]
            if len(greedy_actions) == 0:
                return first_greedy_action
            else:
                return np.random.choice(greedy_actions)

    def sample_reward(true_q, action):
        assert 0 <= action < NUM_ACTIONS
        # Sample the reward from a normal distribution with mean q*(action) and variance 1.0
        return np.random.normal(true_q[action], 1.0)

    # We keep track of all rewards obtained at each time step for all runs.
    rewards = np.zeros(shape=(num_runs, num_time_steps))
    optimal_actions_counter = np.zeros(shape=(num_runs, num_time_steps))

    for r in range(num_runs):
        # At each run, we generate a new true q function, q*, and we estimate the true q function again.
        # At the end of all runs, we average the obtained rewards for each time step, across all runs.

        # Generate the true expected value for each action by sampling from a normal distribution with mean 0 and variance 1.
        # true_q is used to generate the reward at each time step (see below).
        true_q = np.random.normal(0.0, 1.0, size=(NUM_ACTIONS,))

        # We need to keep track of the number of times each action was taken.
        actions_count = np.zeros(shape=(NUM_ACTIONS,), dtype=np.int)

        # We need to keep track of the sum of rewards that were received for each action so far.
        rewards_sum = np.zeros(shape=(NUM_ACTIONS,))

        # Initial estimate of the q function.
        estimated_q = np.zeros(shape=(NUM_ACTIONS,))

        for t in range(num_time_steps):
            action = sample_action(estimated_q)  # action at time step t

            # Check if we selected the optimal action (according to the true_q).
            if action == get_optimal_action(true_q):
                optimal_actions_counter[r, t] += 1

            reward = sample_reward(true_q, action)  # reward at time step t
            rewards[r, t] = reward

            actions_count[action] += 1
            rewards_sum[action] += reward

            estimated_q[action] = rewards_sum[action] / actions_count[action]

        # print("estimated_q =", estimated_q)
        # print("true q =", true_q)

    # Average rewards at each time step across different runs (i.e. bandit problems, aka true q functions)
    average_rewards = np.mean(rewards, axis=0)
    average_optimal_action_counter = np.mean(optimal_actions_counter, axis=0)

    return average_rewards, average_optimal_action_counter


def experiment():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    import matplotlib.gridspec as gridspec
    grid = gridspec.GridSpec(1, 2)

    for epsilon in EPSILONS:
        average_rewards, average_optimal_action_counter = sample_average(epsilon=epsilon)
        ax1.plot(list(range(NUM_TIME_STEPS)), average_rewards)

        ax2.plot(list(range(NUM_TIME_STEPS)), average_optimal_action_counter * 100)

    ax1.set_ylabel('Reward')
    ax1.set_xlabel("Time Step")
    ax1.legend(EPSILONS)

    ax2.set_ylabel('Optimal Action (%)')
    ax2.set_xlabel("Time Step")
    ax2.legend(EPSILONS)

    # plt.suptitle("Results averaged across {} runs\n(i.e. different true q values)".format(NUM_RUNS))

    plt.show()



if __name__ == '__main__':
    experiment()
