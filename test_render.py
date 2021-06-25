"""Test gym environment render."""

import pyvirtualdisplay
from tf_agents.environments import suite_gym

display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))

env_name = 'CartPole-v0'
train_py_env = suite_gym.load(env_name)
train_py_env.reset()
train_py_env.render()
