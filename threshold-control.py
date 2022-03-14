#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 14:46:10 2021

@author: ocail
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import random, os
import datetime
from veh_env import vehEnv
import argparse
import scipy


parser = argparse.ArgumentParser(description='PyTorch actor-critic')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=0, metavar='N',
                    help='random seed (default: 0)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--rho', type=float, default=0,
                    help='rho (default: 0)')
parser.add_argument('--max_time', type=int, default=20,
                    help='max_time (default: 20)')
args = parser.parse_args()


print("Random Seed: ", args.seed)
random.seed(args.seed)

env = vehEnv(T=args.max_time, rho=args.rho)
env.reset()
done = False

# Create a folders for logs
dir_name = 'runs/Threshold_Based' + '_rho_' + str(args.rho) \
           + '_t_' + datetime.datetime.now().strftime("%b-%d_%H_%M_%S")

# save the trained model
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

if not os.path.exists(dir_name + '/test_results'):
    os.mkdir(dir_name + '/test_results')


def th_based(x, th=0.01):
    if (x[1] - 4 * np.sin(2 * np.pi / 50 * x[0])) ** 2 > th:
        action = 1.0
    else:
        action = 0.0
    return action


current_state = torch.tensor(env.x0)
x = torch.unsqueeze(torch.cat((torch.tensor(env.x0), torch.tensor(env.x0)), 0), 0)
r = 0
na = 0
acts = []
jmpcs = []

while not done:
    # if random.randint(0,9)>=6:
    #     act = 1.0
    # else:
    #     act = 0.0
    """change if we want to trigger MPC all the time"""
    act = 1.0 #th_based(current_state, th=0.1)
    acts.append(act)
    # act = 1
    next_state, reward, done, (t, jmpc) = env.step(act)
    x = torch.cat((x, torch.unsqueeze(torch.tensor(next_state), 0)), 0)
    r += reward.item()
    current_state = next_state
    na += act  # TODO: calculate freq?
    jmpcs.append(jmpc)

e = (x[:, 1] - 4 * np.sin(2 * np.pi / 50 * x[:, 0]))
plt.plot(x[:, 0], e)
plt.xlabel('Training epochs')
plt.ylabel('Lateral Errors')
plt.show()
print(torch.abs(e).mean())


plt.figure(1)
plt.title('Tracking Performance')
plt.xlabel('Time steps')
plt.ylabel('lateral position')
plt.xlim([0, len(x[:, 1])])

plt.plot(x[:, 1], label='rl')
plt.plot(4 * np.sin(2 * np.pi / 50 * x[:, 0]), label='y')
plt.legend()
plt.show()

plt.figure(2)
plt.title('Tracking Performance')
plt.xlabel('Time steps')
plt.ylabel('lateral error')
plt.xlim([0, len(x[:, 1])])
plt.plot(x[:, 1] - 4 * np.sin(2 * np.pi / 50 * x[:, 0]), label='error')
plt.legend()
plt.show()

savemat = {
           "rl": list(x[:, 1]),
           "gt": list(4 * np.sin(2 * np.pi / 50 * x[:, 0])),
           "act": acts,
           "return": r,
           "jmpcs": jmpcs
           }

scipy.io.savemat(dir_name + '/test_results/' + 'test_results.mat', savemat)
print(r)