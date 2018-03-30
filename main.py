from Agent import Agent
from Monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v2')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)
print("\n", avg_rewards.pop(), " <> ", best_avg_reward)
