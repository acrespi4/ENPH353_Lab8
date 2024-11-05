#!/usr/bin/env python3
import gym
from gym import wrappers
import gym_gazebo
import time
import random
import time
import liveplot
import os.path
from os import path
import inspect
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim

HIDDEN_SIZE = 128  # number of neurons in the hidden layer
BATCH_SIZE = 16    # number of episodes to play for every network iteration
PERCENTILE = 70    # only the episodes with the top 30% total reward are used for training

class Net(nn.Module):
    '''
    @brief Takes an observation from the environment and outputs a probability 
           for each action we can take.
    '''
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        # Define the NN architecture
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# Stores the total reward for the episode and the steps taken in the episode
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
# Stores the observation and the action the agent took
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

def iterate_batches(env, net, batch_size):
    '''
    @brief A generator function that generates batches of episodes used to train the NN
    @param env: environment handler - allows us to reset and step the simulation
    @param net: neural network we use to predict the next action
    @param batch_size: number of episodes to compile
    @retval batch: returns a batch of batch_size episodes
    '''
    batch = []  # a list of episodes
    episode_reward = 0.0  # current episode total reward
    episode_steps = []  # list of current episode steps
    sm = nn.Softmax(dim=1)  # SOFTMAX object - converts raw NN outputs to probabilities

    # Reset environment and obtain the first observation
    obs = env.reset()

    while True:
        # Convert the observation to a tensor that we can pass into the NN
        obs_v = torch.FloatTensor([obs])

        # Run the NN and convert its output to probabilities
        act_probs_v = sm(net(obs_v))
        # Extract the probabilities associated with each action
        act_probs = act_probs_v.data.numpy()[0]
        # Sample the probability distribution to choose the next action
        action = np.random.choice(len(act_probs), p=act_probs)

        # Run one simulation step using the chosen action
        next_obs, reward, is_done, _ = env.step(action)

        # Add the current step reward to the total episode reward
        episode_reward += reward
        # Add the initial observation and action to the current episode steps
        episode_steps.append(EpisodeStep(observation=obs, action=action))

        # If the episode is done, save it and reset for the next one
        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []

        obs = next_obs  # update observation for the next step

def filter_batch(batch, percentile):
    '''
    @brief Filters out the top percentile of episodes based on rewards
    @param batch: batch of episodes
    @param percentile: percentile threshold for elite episodes
    @retval train_obs_v: observations from elite episodes
    @retval train_act_v: actions from elite episodes
    @retval reward_bound: threshold reward for elite episodes
    @retval reward_mean: mean reward of the batch
    '''
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean

if __name__ == '__main__':
    # Setup environment
    env = gym.make('GazeboCartPole-v0')
    obs_size = env.observation_space.shape[0]  # Set observation size space
    n_actions = env.action_space.n  # Set action size space

    # Create the NN object
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-cartpole")

    # Train the network
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()

        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)

        # Check if the problem is solved
        REWARD_THRESHOLD = 600
        if reward_m > REWARD_THRESHOLD:
            print("Solved!")
            break
    writer.close()
