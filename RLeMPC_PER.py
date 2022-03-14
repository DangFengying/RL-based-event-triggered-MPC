import math, random

import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from veh_env import vehEnv
import csv
import os, datetime

USE_CUDA = False
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs)


class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        states = np.concatenate(batch[0])
        actions = batch[1]
        rewards = batch[2]
        next_states = np.concatenate(batch[3])
        dones = batch[4]

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 5000

num_frames = 50000  # number of steps
batch_size = 64
gamma = 0.99
buffer_size = 10000
# parameter
rho_c = 0.01

eval_per_train = 5
beta_start = 0.4
# parameter
beta_frames = 30000
beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

plt.plot([beta_by_frame(i) for i in range(num_frames)])
#plt.show()

env = vehEnv(T=20, rho=rho_c)
env_test = vehEnv(T=20, rho=rho_c)
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
plt.plot([epsilon_by_frame(i) for i in range(num_frames)])
#plt.show()


class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(env.obs_space, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0]
            action = action.item()
        else:
            action = random.randrange(env.action_space)
        return action


current_model = DQN(env.obs_space, env.action_space)
target_model = DQN(env.obs_space, env.action_space)

if USE_CUDA:
    current_model = current_model.cuda()
    target_model = target_model.cuda()

optimizer = optim.Adam(current_model.parameters())
replay_buffer = NaivePrioritizedBuffer(buffer_size)


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

update_target(current_model, target_model)


def compute_td_loss(batch_size, beta):
    state, action, reward, next_state, done, indices, weights = replay_buffer.sample(batch_size, beta)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))
    weights = Variable(torch.FloatTensor(weights))

    #############original code: DQN###################
    # q_values = current_model(state)
    # next_q_values = target_model(next_state)
    #
    # q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    # next_q_value = next_q_values.max(1)[0]
    # expected_q_value = reward + gamma * next_q_value * (1 - done)

    #############changed: DDQN#######################
    # calculate q
    q_values = current_model(state)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

    # calculate q_target
    q2 = current_model(next_state)
    q_target = target_model(next_state)
    q_target = q_target.gather(1, q2.max(1)[1].unsqueeze(1))
    expected_q_value = reward + gamma * (1 - done) * q_target.max(1)[0]
    ##############change end#######################

    loss = (q_value - expected_q_value.detach()).pow(2) * weights
    prios = loss + 1e-5
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
    optimizer.step()

    return loss

#############################################
# Create a folders for tensorboard, logs
dir_name = 'runs/' + 'DDQN_PER' \
           + '_betaF_' + str(beta_frames) \
           + '_beta0_' + str(beta_start) \
           + '_rho_' + str(rho_c)\
# \
#            + '_t_' + datetime.datetime.now().strftime("%b-%d_%H_%M_%S")
# + '_alpha_' + str(self.prob_alpha) \
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

if not os.path.isfile(dir_name + '/train_logs.csv'):
    with open(dir_name + '/train_logs.csv', mode='w') as csv_file:
        fieldnames = ['Model', 'Epochs', 'max_step', 'Training Return', 'Evaluation Return', "Evaluation Jmpc",
                      'Mean Error', 'Min_from_15t', 'Max_from_15t', 'action_freq']
        writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer_csv.writeheader()

########################################################################
losses = []
all_rewards = []
episode_reward = 0
episode = 0

state = env.reset()
for frame_idx in range(1, num_frames + 1):
    epsilon = epsilon_by_frame(frame_idx)
    action = current_model.act(state, epsilon)

    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)

    state = next_state
    episode_reward += reward

    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        episode += 1

        if (episode + 1) % eval_per_train == 0:
            eval_episode_return = 0
            state_test = env_test.reset()
            done_test = False

            x = [state_test[0]]
            rl = [state_test[1]]  # rl output
            gt = [4 * np.sin(2 * np.pi / 50 * state_test[0])]  # ground truth
            act = []
            jmpcs = []

            while not done_test:
                action_test = current_model.act(state_test, 0)
                state_test, reward_test, done_test, (t, jmpc) = env_test.step(action_test)

                act.append(action_test)
                x.append(state_test[0])
                rl.append(state_test[1])
                gt.append(4 * np.sin(2 * np.pi / 50 * state_test[0]))
                jmpcs.append(jmpc)
                eval_episode_return += reward_test

            max_step = len(rl)
            eval_error = np.array(rl[:max_step]) - 4 * np.sin(2 * np.pi / 50 * np.array(x[:max_step]))

            with open(dir_name + '/train_logs.csv', 'a+', newline='') as write_obj:
                csv_writer = csv.writer(write_obj)
                try:
                    csv_writer.writerow(['dqn_per', episode + 1, max_step, '{:.3f}'.format(all_rewards[-1]),
                                         '{:.3f}'.format(eval_episode_return), '{:.3f}'.format(sum(jmpcs)),
                                         '{:.3f}'.format(np.mean(np.absolute(eval_error))),
                                         '{:.3f}'.format(np.min(eval_error[15:])),
                                         '{:.3f}'.format(np.max(eval_error[15:])),
                                         '{:.3f}'.format(sum(act[:max_step]) / max_step)])
                except:
                    csv_writer.writerow(
                        ['dqn_per', episode + 1, max_step, '{:.3f}'.format(all_rewards[-1]),
                         '{:.3f}'.format(eval_episode_return), '{:.3f}'.format(sum(jmpcs)),
                         '{:.3f}'.format(np.mean(np.absolute(eval_error))),
                         '{:.3f}'.format(np.min(eval_error)), '{:.3f}'.format(np.max(eval_error)),
                         '{:.3f}'.format(sum(act[:max_step]) / max_step)])

    if len(replay_buffer) > batch_size:
        beta = beta_by_frame(frame_idx)
        loss = compute_td_loss(batch_size, beta)
        losses.append(loss.item())

    if frame_idx % 1000 == 0:
        update_target(current_model, target_model)

# save epoch returns
np.save(dir_name + '/epoch_returns.npy', all_rewards)