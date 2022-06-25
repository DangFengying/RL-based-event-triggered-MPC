from abc import ABC, abstractmethod
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ..memory import LazyMultiStepMemory, LazyPrioritizedMultiStepMemory
from ..utils import update_params, RunningMeanStats

import csv
import matplotlib.pyplot as plt
from scipy.io import savemat

def smooth(x, timestamps=2):
    # last 100
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - timestamps)
        y[i] = float(x[start:(i + 1)].sum()) / (i - start + 1)
    return y


class BaseAgent(ABC):
    def __init__(self, env, test_env, log_dir, num_steps=100000, batch_size=64,
                 memory_size=1000000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=20000,
                 update_interval=4, target_update_interval=8000,
                 use_per=False, num_eval_steps=125000, max_episode_steps=27000,
                 log_interval=10, eval_interval=1000, cuda=True, seed=0):
        super().__init__()
        self.env = env
        self.test_env = test_env

        # Set seed.
        torch.manual_seed(seed)
        np.random.seed(seed)
        # torch.backends.cudnn.deterministic = True  # It harms a performance.
        # torch.backends.cudnn.benchmark = False  # It harms a performance.

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        # LazyMemory efficiently stores FrameStacked states.
        if use_per:
            beta_steps = (num_steps - start_steps) / update_interval
            self.memory = LazyPrioritizedMultiStepMemory(
                capacity=memory_size,
                state_shape=self.env.obs_space,
                device=self.device, gamma=gamma, multi_step=multi_step,
                beta_steps=beta_steps)
        else:
            self.memory = LazyMultiStepMemory(
                capacity=memory_size,
                state_shape=self.env.obs_space,
                device=self.device, gamma=gamma, multi_step=multi_step)

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_return = RunningMeanStats(log_interval)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.best_eval_score = -np.inf
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.gamma_n = gamma ** multi_step
        self.start_steps = start_steps
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.use_per = use_per
        self.num_eval_steps = num_eval_steps
        self.max_episode_steps = max_episode_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval

        if not os.path.isfile(self.log_dir + '/train_logs.csv'):
            with open(self.log_dir + '/train_logs.csv', mode='w') as csv_file:
                fieldnames = ['Model', 'Epochs', 'max_step', 'Training Return', 'Evaluation Return', "Evaluation Jmpc",
                              'Mean Error', 'Min_from_15t', 'Max_from_15t', 'action_freq']
                writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer_csv.writeheader()

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return self.steps % self.update_interval == 0\
            and self.steps >= self.start_steps

    @abstractmethod
    def explore(self, state):
        pass

    @abstractmethod
    def exploit(self, state):
        pass

    @abstractmethod
    def update_target(self):
        pass

    @abstractmethod
    def calc_current_q(self, states, actions, rewards, next_states, dones):
        pass

    @abstractmethod
    def calc_target_q(self, states, actions, rewards, next_states, dones):
        pass

    @abstractmethod
    def calc_critic_loss(self, batch, weights):
        pass

    @abstractmethod
    def calc_policy_loss(self, batch, weights):
        pass

    @abstractmethod
    def calc_entropy_loss(self, entropies, weights):
        pass

    def train_episode(self):
        self.episodes += 1
        episode_return = 0.
        episode_steps = 0

        done = False
        state = self.env.reset()

        while (not done) and episode_steps <= self.max_episode_steps:
            if self.start_steps > self.steps:
                action = np.random.randint(self.env.action_space)
            else:
                action = self.explore(state)

            next_state, reward, done, _ = self.env.step(action)

            # Clip reward to [-1.0, 1.0].
            clipped_reward = max(min(reward, 1.0), -1.0)
            # clipped_reward = reward / 10

            # To calculate efficiently, set priority=max_priority here.
            self.memory.append(state, action, clipped_reward, next_state, done)

            self.steps += 1
            episode_steps += 1
            episode_return += reward
            state = next_state

            if self.is_update():
                self.learn()

            if self.steps % self.target_update_interval == 0:
                self.update_target()

        if self.episodes % self.eval_interval == 0:
            self.evaluate()
            self.save_models(os.path.join(self.model_dir, 'model' + str(self.episodes)))

        # We log running mean of training rewards.
        self.train_return.append(episode_return)

        if self.episodes % self.log_interval == 0:
            self.writer.add_scalar(
                'reward/train_return', episode_return, self.episodes)
            self.writer.add_scalar(
                'reward/train_steps', episode_steps, self.episodes)

    def learn(self):
        assert hasattr(self, 'q1_optim') and hasattr(self, 'q2_optim') and\
            hasattr(self, 'policy_optim') and hasattr(self, 'alpha_optim')

        self.learning_steps += 1

        if self.use_per:
            batch, weights = self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)
            # Set priority weights to 1 when we don't use PER.
            weights = 1.

        q1_loss, q2_loss, errors, mean_q1, mean_q2 = \
            self.calc_critic_loss(batch, weights)
        policy_loss, entropies = self.calc_policy_loss(batch, weights)
        entropy_loss = self.calc_entropy_loss(entropies, weights)

        update_params(self.q1_optim, q1_loss)
        update_params(self.q2_optim, q2_loss)
        update_params(self.policy_optim, policy_loss)
        update_params(self.alpha_optim, entropy_loss)

        self.alpha = self.log_alpha.exp()

        if self.use_per:
            self.memory.update_priority(errors)

        if self.learning_steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/Q1', q1_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/Q2', q2_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/policy', policy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/alpha', entropy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/alpha', self.alpha.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q1', mean_q1, self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q2', mean_q2, self.learning_steps)
            self.writer.add_scalar(
                'stats/entropy', entropies.detach().mean().item(),
                self.learning_steps)

    def evaluate(self):
        num_steps = 0
        state = self.test_env.reset()
        episode_return = 0.0
        done = False

        # save logs
        x = [state[0]]
        rl = [state[2]]  # rl output
        gt = [4 * np.sin(2 * np.pi / 50 * state[0])]  # ground truth
        act = []
        jmpcs = []

        while not done:
            action = self.exploit(state)
            next_state, reward, done, (t, jmpc) = self.test_env.step(action)
            num_steps += 1
            episode_return += reward
            state = next_state

            act.append(action)
            x.append(next_state[0])
            rl.append(next_state[2])
            gt.append(4 * np.sin(2 * np.pi / 50 * next_state[0]))
            jmpcs.append(jmpc)

        self.writer.add_scalar(
            'reward/test_return', episode_return, self.episodes)
        self.writer.add_scalar(
            'reward/test_steps', num_steps, self.episodes)

        max_step = len(rl)
        eval_error = np.array(rl[:max_step]) - 4 * np.sin(2 * np.pi / 50 * np.array(x[:max_step]))

        save_mat = {
            'act': act,
            'x': x,
            'rl': rl,
            'gt': gt,
            'jmpcs': jmpcs
        }
        if not os.path.exists(self.log_dir + '/results'):
            os.mkdir(self.log_dir + '/results')

        savemat(self.log_dir + '/results/a2c_results_{}.mat'.format(self.episodes), save_mat)

        with open(self.log_dir + '/train_logs.csv', 'a+', newline='') as write_obj:
            csv_writer = csv.writer(write_obj)
            try:
                csv_writer.writerow(['SAC', self.episodes, max_step, '{:.3f}'.format(self.train_return.get()),
                                     '{:.3f}'.format(episode_return), '{:.3f}'.format(sum(jmpcs)),
                                     '{:.3f}'.format(np.mean(np.absolute(eval_error))),
                                     '{:.3f}'.format(np.min(eval_error[15:])),
                                     '{:.3f}'.format(np.max(eval_error[15:])),
                                     '{:.3f}'.format(sum(act[:max_step]) / max_step)])
            except:
                csv_writer.writerow(
                    ['SAC', self.episodes, max_step, '{:.3f}'.format(self.train_return.get()),
                     '{:.3f}'.format(episode_return), '{:.3f}'.format(sum(jmpcs)),
                     '{:.3f}'.format(np.mean(np.absolute(eval_error))),
                     '{:.3f}'.format(np.min(eval_error)), '{:.3f}'.format(np.max(eval_error)),
                     '{:.3f}'.format(sum(act[:max_step]) / max_step)])

    def test(self):
        state = self.test_env.reset()
        episode_return = 0.0
        done = False

        # save logs
        x = [state[0]]
        rl = [state[2]]  # rl output
        gt = [4 * np.sin(2 * np.pi / 50 * state[0])]  # ground truth
        act = []
        jmpcs = []

        while not done:
            action = self.explore(state)
            next_state, reward, done, (t, jmpc) = self.test_env.step(action)
            episode_return += reward
            state = next_state

            act.append(action)
            x.append(next_state[0])
            rl.append(next_state[2])
            gt.append(4 * np.sin(2 * np.pi / 50 * next_state[0]))
            jmpcs.append(jmpc)

        print("episode reward", episode_return)
        plt.figure(1)
        plt.title('Tracking Performance')
        plt.xlabel('Time (s)')
        plt.ylabel('lateral')
        xi = np.arange(0, 20, 0.2)
        plt.plot(xi, rl[:100], label='rl')
        plt.plot(xi, gt[:100], label='gt')
        plt.legend()
        plt.show()

        plt.figure(2)
        plt.title('Tracking Performance')
        plt.xlabel('Time (s)')
        plt.ylabel('lateral error')
        xi = np.arange(0, 20, 0.2)
        plt.plot(xi, smooth(np.array(rl[:100]) - 4 * np.sin(2 * np.pi / 50 * np.array(x[:100]))), label='error')
        plt.legend()
        plt.show()

        print("action ratio", sum(act) / len(act))
        plt.figure(3)
        plt.title('Actions')
        plt.xlabel('Time (s)')
        plt.ylabel('Action')
        xi = np.arange(0, 20, 0.2)
        plt.scatter(xi, act[:100])
        plt.plot(xi, smooth(np.array(act[:100])))
        plt.show()

    @abstractmethod
    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def __del__(self):
        self.env.close()
        self.test_env.close()
        self.writer.close()
