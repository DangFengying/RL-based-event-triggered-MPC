# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 21:24:01 2021

@author: junchen
"""

from veh_env import vehEnv
# import numpy as np
# import math
import matplotlib
import matplotlib.pyplot as plt
import random
import torch
# import pandas as pd
import pickle
import datetime
import argparse
import os
# from RL_lib import Transition
from RL_lib import LinQ
from RL_lib import ReplayMemory
from RL_lib import LinRLagent
import numpy as np
import json

torch.set_printoptions(precision=3)
import scipy, csv

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
parser.add_argument('--num_episodes', type=int, default=200,
                    help='max_time (default: 200)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed + 1)

# # set up matplotlib
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
#     from IPython import display
#
# plt.ion()

# if gpu is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Connect to environment
env = vehEnv(T=args.max_time, rho=args.rho)
env_eval = vehEnv(T=args.max_time, rho=args.rho)
# Get number of actions and states from environment
n_states = 10

# Create a folders for logs
dir_name = 'runs/LinearQ' + '_rho_' + str(args.rho) \
           + '_t_' + datetime.datetime.now().strftime("%b-%d_%H_%M_%S")

# save the trained model
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

if not os.path.exists(dir_name + '/save_model'):
    os.mkdir(dir_name + '/save_model')

if not os.path.exists(dir_name + '/test_results'):
    os.mkdir(dir_name + '/test_results')

argu = {
        "rho": args.rho,
        "algo": "LinearQ",
        "date": datetime.datetime.now().strftime("%b-%d_%H_%M_%S"),
        "prediction_horizon": env.p,
        "env_threshold_error": env.threshold_error
    }

with open(dir_name + '/' + "arguments.json", "w") as outfile:
    json.dump(argu, outfile)

params = {'device': device,
          'batch_size': 32,
          'gamma': 1.0,
          'eps_start': 0.5,
          'eps_end': 0.00,
          'eps_end_step': args.num_episodes // 2,
          'num_episodes': args.num_episodes}  # 500}

state = torch.tensor(env.reset())
done = False

RL = LinRLagent(policy_net=LinQ(n_states=n_states), ReplayMemory=ReplayMemory(5000), params=params,
                target_net=LinQ(n_states=n_states), behavior_net=LinQ(n_states=n_states))
RL.set_action_space(torch.tensor([0, 1]))  # set action space
n_feat = RL.get_feat(state, torch.tensor(0)).size()[0]
RL.policy_net.reset_weights(torch.randn(n_feat, dtype=torch.float64))
RL.update_target_net()
RL.update_behavior_net()

if not os.path.isfile(dir_name + '/train_logs.csv'):
    with open(dir_name + '/train_logs.csv', mode='w') as csv_file:
        fieldnames = ['Model', 'Epochs', 'max_step', 'Training Return', 'Evaluation Return', "Evaluation Jmpc",
                      'Mean Error', 'Min_from_15t', 'Max_from_15t', 'action_freq']
        writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer_csv.writeheader()

# def plot_training_return(r, i_episode, fig_idx=1, len_to_average=10):
#     if i_episode < 1:
#         return
#
#     r = r[0:i_episode + 1]
#     plt.figure(fig_idx)
#     plt.clf()
#
#     plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('G')
#     plt.plot(r)
#
#     # Take some episode averages and plot them too
#     if len(r) >= len_to_average:
#         means = r.unfold(0, len_to_average, 1).mean(1).view(-1)
#         means = torch.cat((torch.zeros(len_to_average - 1), means))
#         plt.plot(means)
#
#     plt.pause(0.001)  # pause a bit so that plots are updated
#     if is_ipython:
#         display.clear_output(wait=True)
#         display.display(plt.gcf())
#     plt.savefig("training.png")

RL.w_hist = torch.zeros(args.num_episodes, n_feat, dtype=torch.float64)

for i_episode in range(args.num_episodes):
    # Initialize the environment and state
    step_rewards = []
    total_num_act = 0
    maxX = torch.tensor([-1.0])
    state = env.reset()
    done = False
    state = torch.tensor(state)
    # state, _, done, _ = env.step(0)
    explore = RL.explore(step_adv=True)

    while not done:
        # Select and perform an action
        explore = RL.explore(step_adv=False)
        action = RL.set_action(state, explore=explore, net_opt='behavior')

        next_state, reward, done, (t, jmpc) = env.step(action.item())
        next_state = torch.tensor(next_state)
        reward = torch.tensor([reward], device=device)
        step_rewards.append(reward)
        total_num_act += action.item()

        # # Store the transition in memory
        RL.ReplayMemory.push(state, action, next_state, reward)

        # # Move to the next state
        state = next_state

        # # Perform one step of the optimization (on the target network)        
        RL.optimize_model()
        RL.update_target_net()

    r = torch.tensor(step_rewards).sum()
    RL.log(i_episode, r, net_opt='behavior')
    total_num_act = max(total_num_act, 1)
    print('Epoch = {}; return = {:.5f}; Ts = {:.5f}; w = {}'.format(i_episode, r.item(), t.item() / total_num_act,
                                                                    RL.behavior_net.w))

    # plot_training_return(RL.return_epoch, i_episode)

    RL.update_behavior_net()
    if i_episode % 20 == 0:
        RL.update_behavior_net(RL.w_hist[RL.return_best_epoch, :])
        RL.update_target_net(RL.w_hist[RL.return_best_epoch, :])

    if i_episode % 1 == 0:
        # Do an evaluation
        eval_episode_return = 0
        state_eval = env_eval.reset()
        done_eval = False
        state_eval = torch.tensor(state_eval)

        # environmental logs
        x = [state_eval[0]]
        rl = [state_eval[1]]  # rl output
        gt = [4 * np.sin(2 * np.pi / 50 * state_eval[0])]  # ground truth
        act = []
        jmpcs = []

        while not done_eval:
            action_eval = RL.set_action(state_eval, explore=False, net_opt='behavior')
            next_state_eval, reward_eval, done_eval, (t_eval, jmpc) = env_eval.step(action_eval.item())
            next_state_eval = torch.tensor(next_state_eval)
            eval_episode_return += reward_eval

            # evaluation logs
            act.append(action_eval)
            x.append(next_state_eval[0])
            rl.append(next_state_eval[1])
            gt.append(4 * np.sin(2 * np.pi / 50 * next_state_eval[0]))
            jmpcs.append(jmpc)

            state_eval = next_state_eval

        savemat = {
            "x": x,
            "rl": rl,
            "gt": gt,
            "act": act,
            "jmpcs": jmpcs
        }

        max_step = len(rl)
        print(max_step)
        eval_error = np.array(rl[:max_step]) - 4 * np.sin(2 * np.pi / 50 * np.array(x[:max_step]))

        scipy.io.savemat(dir_name + '/test_results/' + 'test_results_' + str(i_episode) + '.mat', savemat)
        torch.save(RL.behavior_net.w, dir_name + '/save_model/w_' + str(i_episode) + '.pt')
        np.save(dir_name + '/epoch_returns.npy', RL.return_epoch)

        # save test results
        with open(dir_name + '/train_logs.csv', 'a+', newline='') as write_obj:
            csv_writer = csv.writer(write_obj)
            try:
                csv_writer.writerow(
                    ["LinearQ", i_episode, max_step, '{:.3f}'.format(RL.return_epoch[i_episode]),
                     '{:.3f}'.format(eval_episode_return), '{:.3f}'.format(eval_episode_return),
                     '{:.3f}'.format(np.mean(np.absolute(eval_error))),
                     '{:.3f}'.format(np.min(eval_error)), '{:.3f}'.format(np.max(eval_error)),
                     '{:.3f}'.format(sum(act[:max_step]) / max_step)])
            except:
                csv_writer.writerow(
                    ["LinearQ", i_episode, max_step, '{:.3f}'.format(RL.return_epoch[i_episode]),
                     '{:.3f}'.format(eval_episode_return), '{:.3f}'.format(eval_episode_return),
                     '{:.3f}'.format(np.mean(np.absolute(eval_error))),
                     '{:.3f}'.format(np.min(eval_error)), '{:.3f}'.format(np.max(eval_error)),
                     '{:.3f}'.format(sum(act[:max_step]) / max_step)])
# with open('RL_empc_AV_{}.pkl'.format(rho), 'wb') as output:
#     pickle.dump(RL, output, pickle.HIGHEST_PROTOCOL)


env.close()
