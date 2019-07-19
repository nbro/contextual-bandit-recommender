import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.mushroom.sample_data import sample_mushroom
from datasets.preprocessing import load_data
from models.context_free_policy import EpsilonGreedyPolicy, RandomPolicy, SampleMeanPolicy, UCBPolicy
from models.context_based_policy import LinUCBPolicy, LinUCBHybridPolicy, LinearRegressorPolicy
from simulate import simulate_contextual_bandit
