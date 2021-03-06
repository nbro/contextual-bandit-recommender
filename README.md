[![Build Status](https://travis-ci.com/dhfromkorea/contextual-bandit-recommender.svg?token=LpCqnxSYFM2Cg2x3ixjz&branch=master)](https://travis-ci.com/dhfromkorea/contextual-bandit-recommender)

# Context-free and contextual bandit algorithms for item recommendation

## How to download the dataset?

Before training any context-free or contextual bandit agent, you need to download the dataset that is used to train it. Specifically, we will use the [mushrooms](https://archive.ics.uci.edu/ml/datasets/mushroom) dataset, which can be downloaded as follows (provided you have `curl` installed)

```
curl https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data -o ./datautils/mushroom/mushroom.csv
```

So, the mushrooms dataset will be downloaded into the folder `./datautils/mushroom/` and it will be called `./datautils/mushroom/mushroom.csv`. You can find more information about this dataset at https://archive.ics.uci.edu/ml/datasets/mushroom.

## How to install the required packages?

To run the experiments, you need to install a few required packages, which you can do as follows

    pip install -r requirements.txt

## How to run the experiments?

To run the experiments on the mushrooms dataset, you can execute the following command

    python main.py "mushroom" --n_trials 10 --n_rounds 1000
    
where `n_rounds` is the number of time steps (iterations) that the bandit algorithm runs for and `n_trials` is the number of times the same experiments is run for.

The results of the experiments are written to the folder `results/`.

## Hyper-parameters

In the [`main.py`](./main.py), you can see that there are other hyper-parameters that can be set or re-set.


## How to plot the results?

To plot the results, run the following command

    python evaluations/plotting.py "mushroom" --n_trials 10 --window 10

## Available algorithms/agents

### Context-free bandits

* Epsilon Greedy
* UCB policy
* Sample Mean Policy
* Random Policy

### Contextual bandit algorithms

* LinUCB: algorithm 1 and a modified version of algorithm 2 in the paper [1].
* Thompson Sampling: Linear Gaussian with a conjugate prior [2].
* Neural Network Policy: A fully-connected neural network with gradient noise.

## Results

Here is an example of the plots of the results of an experiment on the mushrooms dataset.

![Mushroom Cumaltive Regret](http://www.dhfromkorea.com/images/cb/mushroom.cumreg.png)

![Mushroom Action Distribution](http://www.dhfromkorea.com/images/cb/mushroom.acts.png)

## Terminology

- [Click-through rate](https://en.wikipedia.org/wiki/Click-through_rate)
- Regret
- Reward

## Other resources

- Serious implementations of bandit algorithms: https://github.com/tensorflow/agents/tree/master/tf_agents/bandits.


[1]: http://rob.schapire.net/papers/www10.pdf
[2]: http://proceedings.mlr.press/v28/agrawal13.pdf


