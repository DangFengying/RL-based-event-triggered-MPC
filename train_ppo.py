import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from veh_env2 import vehEnv
from PPO.config import AgentConfig
from PPO.network import MlpPolicy
import datetime
import random
import os
import csv
from scipy.io import savemat

# Set a random seed
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cpu")
TOTAL_EPISODES = 500


class Agent(AgentConfig):
    def __init__(self, dir, writer, rho_c=0):
        self.env = vehEnv(T=20, rho=rho_c)
        self.env_test = vehEnv(T=20, rho=rho_c)
        self.action_size = self.env.action_space
        self.policy_network = MlpPolicy(action_size=self.action_size).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.k_epoch,
                                                   gamma=0.999)
        self.loss = 0
        self.writer = writer
        self.dir = dir
        self.criterion = nn.MSELoss()
        self.memory = {
            'state': [], 'action': [], 'reward': [], 'next_state': [], 'action_prob': [], 'terminal': [], 'count': 0,
            'advantage': [], 'td_target': torch.FloatTensor([])}

        if not os.path.isfile(dir + '/train_logs.csv'):
            with open(dir + '/train_logs.csv', mode='w') as csv_file:
                fieldnames = ['Model', 'Epochs', 'max_step', 'Training Return', 'Evaluation Return', "Evaluation Jmpc",
                              'Mean Error', 'Min_from_15t', 'Max_from_15t', 'action_freq']
                self.writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
                self.writer_csv.writeheader()

    def train(self):
        episode = 0
        step = 0

        # A new episode
        for _ in range(TOTAL_EPISODES):
            start_step = step
            episode += 1
            episode_length = 0
            terminal = False

            # Get initial state
            state = self.env.reset()
            current_state = state
            total_episode_reward = 0
            actions = []

            # do one episode
            while not terminal:
                step += 1
                episode_length += 1

                # choose an action
                prob_a = self.policy_network.pi(torch.FloatTensor(current_state).to(device))
                action = torch.distributions.Categorical(prob_a).sample().item()
                actions.append(action)

                # act
                state, reward, terminal, _ = self.env.step(action)
                new_state = state

                self.add_memory(current_state, action, reward, new_state, terminal, prob_a[action].item())

                current_state = new_state
                total_episode_reward += reward

                if terminal:
                    episode_length = step - start_step

                    # add logs
                    self.writer.add_scalar('Train/steps', episode_length, episode)
                    self.writer.add_scalar('Train/EpisodeReturns', total_episode_reward, episode)
                    self.writer.add_scalar('Train/action_ratio', sum(actions) / len(actions), episode)

                    self.finish_path(episode_length)

                    # do a test
                    if episode % self.eval_freq == 0:
                        state_test = self.env_test.reset()
                        terminal_test = False
                        test_steps = 0
                        rewards_test = 0
                        actions_test = []

                        # save logs
                        x = [state_test[0]]
                        rl = [state_test[2]]  # rl output
                        gt = [4 * np.sin(2 * np.pi / 50 * state_test[0])]  # ground truth
                        act = []
                        jmpcs = []

                        while not terminal_test:
                            test_steps += 1
                            prob_a_test = self.policy_network.pi(torch.FloatTensor(state_test).to(device))
                            action_test = torch.distributions.Categorical(prob_a_test).sample().item()
                            state_test, reward_test, terminal_test, (t, jmpc) = self.env_test.step(action_test)
                            rewards_test += reward_test
                            actions_test.append(action_test)

                            act.append(action_test)
                            x.append(state_test[0])
                            rl.append(state_test[2])
                            gt.append(4 * np.sin(2 * np.pi / 50 * state_test[0]))
                            jmpcs.append(jmpc)

                        self.writer.add_scalar('Eval/steps', test_steps, episode)
                        self.writer.add_scalar('Eval/EpisodeReturns', rewards_test, episode)
                        self.writer.add_scalar('Eval/action_ratio', sum(actions_test) / len(actions_test), episode)

                        save_mat = {
                            'act': act,
                            'x': x,
                            'rl': rl,
                            'gt': gt,
                            'jmpcs': jmpcs}
                        if not os.path.exists(self.dir + '/results'):
                            os.mkdir(self.dir + '/results')

                        savemat(self.dir + '/results/ppo_results_{}.mat'.format(episode), save_mat)

                        max_step = len(rl)
                        eval_error = np.array(rl[:max_step]) - 4 * np.sin(2 * np.pi / 50 * np.array(x[:max_step]))

                        with open(self.dir + '/train_logs.csv', 'a+', newline='') as write_obj:
                            csv_writer = csv.writer(write_obj)
                            try:
                                csv_writer.writerow(['PPO', episode, max_step, '{:.3f}'.format(total_episode_reward),
                                                     '{:.3f}'.format(rewards_test), '{:.3f}'.format(sum(jmpcs)),
                                                     '{:.3f}'.format(np.mean(np.absolute(eval_error))),
                                                     '{:.3f}'.format(np.min(eval_error[15:])),
                                                     '{:.3f}'.format(np.max(eval_error[15:])),
                                                     '{:.3f}'.format(sum(actions_test) / len(actions_test))])
                            except:
                                csv_writer.writerow(
                                    ['PPO', episode, max_step, '{:.3f}'.format(total_episode_reward),
                                     '{:.3f}'.format(rewards_test), '{:.3f}'.format(sum(jmpcs)),
                                     '{:.3f}'.format(np.mean(np.absolute(eval_error))),
                                     '{:.3f}'.format(np.min(eval_error)), '{:.3f}'.format(np.max(eval_error)),
                                     '{:.3f}'.format(sum(actions_test) / len(actions_test))])

                    self.env.reset()
                    break

            if episode % self.update_freq == 0:
                for _ in range(self.k_epoch):
                    self.update_network()

        self.env.close()

    def update_network(self):
        # get ratio
        pi = self.policy_network.pi(torch.FloatTensor(self.memory['state']).to(device))
        new_probs_a = torch.gather(pi, 1, torch.tensor(self.memory['action']))
        old_probs_a = torch.FloatTensor(self.memory['action_prob'])
        ratio = torch.exp(torch.log(new_probs_a) - torch.log(old_probs_a))

        # surrogate loss
        surr1 = ratio * torch.FloatTensor(self.memory['advantage'])
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * torch.FloatTensor(self.memory['advantage'])
        pred_v = self.policy_network.v(torch.FloatTensor(self.memory['state']).to(device))
        v_loss = 0.5 * (pred_v - self.memory['td_target']).pow(2)  # Huber loss
        entropy = torch.distributions.Categorical(pi).entropy()
        entropy = torch.tensor([[e] for e in entropy])
        self.loss = (-torch.min(surr1, surr2) + self.v_coef * v_loss - self.entropy_coef * entropy).mean()

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def add_memory(self, s, a, r, next_s, t, prob):
        if self.memory['count'] < self.memory_size:
            self.memory['count'] += 1
        else:
            self.memory['state'] = self.memory['state'][1:]
            self.memory['action'] = self.memory['action'][1:]
            self.memory['reward'] = self.memory['reward'][1:]
            self.memory['next_state'] = self.memory['next_state'][1:]
            self.memory['terminal'] = self.memory['terminal'][1:]
            self.memory['action_prob'] = self.memory['action_prob'][1:]
            self.memory['advantage'] = self.memory['advantage'][1:]
            self.memory['td_target'] = self.memory['td_target'][1:]

        self.memory['state'].append(s)
        self.memory['action'].append([a])
        self.memory['reward'].append([r])
        self.memory['next_state'].append(next_s)
        self.memory['terminal'].append([1 - t])
        self.memory['action_prob'].append(prob)

    def finish_path(self, length):
        state = self.memory['state'][-length:]
        reward = self.memory['reward'][-length:]
        next_state = self.memory['next_state'][-length:]
        terminal = self.memory['terminal'][-length:]

        td_target = torch.FloatTensor(reward) + \
                    self.gamma * self.policy_network.v(torch.FloatTensor(next_state)) * (torch.FloatTensor(terminal))
        delta = td_target - self.policy_network.v(torch.FloatTensor(state))
        delta = delta.detach().numpy()

        # get advantage
        advantages = []
        adv = 0.0
        for d in delta[::-1]:
            adv = self.gamma * self.lmbda * adv + d[0]
            advantages.append([adv])
        advantages.reverse()

        if self.memory['td_target'].shape == torch.Size([1, 0]):
            self.memory['td_target'] = td_target.data
        else:
            self.memory['td_target'] = torch.cat((self.memory['td_target'], td_target.data), dim=0)
        self.memory['advantage'] += advantages


if __name__ == '__main__':
    rho_c = 0
    dir_name = 'runs/PPO' + '_rho_' + str(rho_c) \
               + '_t_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    writer = SummaryWriter(log_dir=dir_name)

    agent = Agent(dir_name, writer, rho_c)
    agent.train()