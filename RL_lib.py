# -*- coding: utf-8 -*-
"""
Created on Thu May 13 12:35:39 2021

@author: junchen
"""


import random
from collections import namedtuple
import torch.nn as nn
import torch
import torch.nn.functional as F
# import math

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))



class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, nHist, nFeat, nOutput):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv1d(nHist, 16, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=5, stride=1)
        self.bn3 = nn.BatchNorm1d(32)

        # Number of Linear input connections depends on output of conv1d layers
        # and therefore the input image size, so compute it.
        def conv1d_size_out(size, kernel_size = 5, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convL = conv1d_size_out(conv1d_size_out(conv1d_size_out(nFeat)))
        linear_input_size = convL * 32
        self.head = nn.Linear(linear_input_size, nOutput)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class LinQ:
    def __init__(self, n_states=1):
        self.w = torch.zeros(n_states, dtype=torch.float64)
        
    def reset_weights(self, w=torch.zeros(1,dtype=torch.float64)):
        self.w = w
        
    def forward(self, x):
        return torch.dot(self.w, x)


class TblQ:
    def __init__(self, n_action = 1, n_states=1, dx=torch.tensor([0.01],dtype=torch.float64),n = 100):
        self.w = torch.zeros([n_action,n+1],dtype=torch.float64)
        self.dx = dx
        self.n_states = n_states
        self.n = torch.tensor(n,dtype=torch.long).unsqueeze(0)
        
    def reset_weights(self,w=torch.zeros(1,dtype=torch.float64)):
        self.w = w
        
    def locate(self,x):
        idx = torch.floor(x/self.dx).int()
        if idx >= self.n:
            idx = self.n
        return idx
    
    def forward(self, x, a):
        idx = self.locate(x)
        if self.n_states == 1:
            return self.w[a,idx[0]]
        else: # only up to 2d table is currently supported
            return self.w[a,idx[0],idx[1]] 


class RLagent:
    def __init__(self, policy_net, ReplayMemory, params, target_net= None, behavior_net = None ):
        self.policy_net = policy_net
        self.target_net = target_net
        self.behavior_net = behavior_net
        self.ReplayMemory = ReplayMemory
        self.device = params['device']
        self.batch_size = params['batch_size'] 
        self.gamma = params['gamma'] 
        self.eps_start = params['eps_start'] 
        self.eps_end = params['eps_end'] 
        self.eps_end_step = params['eps_end_step']
        self.num_episodes = params['num_episodes'] 
        # self.nHist = params['nHist'] 
        self.steps_done = 0
        self.return_epoch = torch.zeros(self.num_episodes)
        self.target_update_count = 0
        self.init_local(params=params)
        self.epsilon = 1.0
        self.epsilon_decay = 0.98
    
    def explore(self, step_adv = True):
        # eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
        #     math.exp(-1. * self.steps_done / self.eps_decay)
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * (self.eps_end_step - self.steps_done) / self.eps_end_step
        # self.epsilon *= self.epsilon_decay
        # self.epsilon = max(self.epsilon, 0.02)

        if step_adv:
            self.steps_done += 1
        if random.random() < eps_threshold:
            return True
        else:
            return False
        
    def set_action_space(self,action_space):
        self.act_space = action_space
        self.n_actions = action_space.numel()
        
    def unpack_memory(self):
        transitions = self.ReplayMemory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        state = batch.state
        action = batch.action
        reward = batch.reward
        next_state = batch.next_state
        
        return state, action, reward, next_state
        
    def optimize_model(self):
        if len(self.ReplayMemory) < self.batch_size:
            return
        
        state, action, reward, next_state = self.unpack_memory()
        self.update_net_weights(state, action, reward, next_state)
        
    def log(self, i_episode, r, net_opt=None):
        self.return_epoch[i_episode] = r
        self.return_best_epoch = torch.argmax(self.return_epoch[0:i_episode+1])
        self.log_weights(i_episode,net_opt=net_opt)

    def update_target_net(self):
        pass
        
    def update_behavior_net(self):
        pass


class LinRLagent(RLagent):
    def init_local(self, params):
        self.w_hist = torch.zeros(self.num_episodes,dtype=torch.float64)
    
    def get_feat(self,x,a):
        # return torch.cat((x*(1.0-a),a.unsqueeze(0)))
        y = torch.zeros(x.size()[0],dtype=torch.float64)
        # y = torch.zeros(x.size, dtype=torch.float64)
        # if a == 1:
        #     return torch.cat((x,torch.tensor([1.0,0.0])))
        # else:
        #     return torch.cat((y,torch.tensor([0.0,1.0])))
        if a == 1:
            return torch.cat((x,y,torch.tensor([1.0,0.0])))
        else:
            return torch.cat((y,x,torch.tensor([0.0,1.0])))
        # if a == 1:
        #     return torch.cat((x,torch.tensor([1.0])))
        # else:
        #     return torch.cat((y,torch.tensor([0.0])))
    
    def set_action(self, x, explore=False, net_opt=None):
        if explore:
            a = self.act_space[torch.randint(self.n_actions, (1,))]
        else:
            Q = torch.tensor([-float('Inf')],dtype=torch.float64)
            a = self.act_space[0]
            for a1 in self.act_space:
                if net_opt == 'target':
                    Q1 = self.target_net.forward(self.get_feat(x,a1))
                elif net_opt == 'behavior':
                    Q1 = self.behavior_net.forward(self.get_feat(x,a1))
                else:
                    Q1 = self.policy_net.forward(self.get_feat(x,a1))
                if Q1 > Q:
                    Q = Q1
                    a = a1                
                
        return a              
    
    def update_net_weights(self, state, action, reward, next_state):
        n = self.policy_net.w.size()[0]
        
        B = 0.00001 * torch.eye(n)
        b = torch.zeros(n)
        for k in range(self.batch_size):
            xsa = self.get_feat(state[k],action[k])
            xsa1 = self.get_feat(next_state[k],self.set_action(next_state[k],net_opt='target'))
            B = B + torch.outer(xsa, xsa-self.gamma*xsa1)
            b = b + xsa*reward[k]
        
        Binv = B.inverse()
        self.policy_net.w = torch.matmul(Binv,b)
        
    def log_weights(self,i_episode,net_opt=None):
        if net_opt == 'target':
            self.w_hist[i_episode,:]=self.target_net.w
        elif net_opt == 'behavior':
            self.w_hist[i_episode,:]=self.behavior_net.w
        else:
            self.w_hist[i_episode,:]=self.policy_net.w
        
    def update_target_net(self, w=None):
        if w is None:
            self.target_net.w = self.policy_net.w
        else:
            self.target_net.w = w
        
    def update_behavior_net(self, w=None):
        if w is None:
            self.behavior_net.w = self.policy_net.w
        else:
            self.behavior_net.w = w


class TblRLagent(RLagent):
    
    def init_local(self,params):
        self.w_hist = torch.zeros(self.num_episodes,dtype=torch.float64)
        self.alpha=params['alpha']
        self.alpha_dec_rate=params['alpha_dec_rate']
        self.ReplayMemory.capacity = 1 # only current observation is used in table case
       
    def set_action(self,x,explore=False,net_opt=None):
        if explore:
            a = self.act_space[torch.randint(self.n_actions,(1,))]
        else:
            Q = torch.tensor([-float('Inf')],dtype=torch.float64)
            a = self.act_space[0]
            for a1 in self.act_space:
                Q1 = self.policy_net.forward(x,a1)
                if Q1>Q:
                    Q = Q1
                    a = a1                
            
        return a              
    
    def update_net_weights(self, state, action, reward, next_state):
        state = state[0]
        action = action[0]
        reward = reward[0]
        next_state = next_state[0]
        a1 = self.set_action(next_state)
        delta = reward+self.gamma*self.policy_net.forward(next_state,a1)-self.policy_net.forward(state,action)
        
        
        idx = self.policy_net.locate(state)
        if self.policy_net.n_states == 1:
            self.policy_net.w[action,idx[0]] += self.alpha*delta.squeeze()
        else: # only up to 2d table is currently supported
            self.policy_net.w[action,idx[0],idx[1]] += self.alpha*delta
        
        
    def log_weights(self,i_episode,net_opt=None):
        self.w_hist[i_episode,:]=self.policy_net.w