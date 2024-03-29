# RL-based-event-triggered-MPC

This is official implementation of our paper: Event-Triggered Model Predictive Control with Deep Reinforcement Learning.

## Installation
- create a anaconda environment via: `conda create -n empc python=3.8 -y`
- activate the virtual env via: `conda activate empc`
- install the requirements via: `pip install -r requirements.txt`

## Algorithms

All the rule-based/RL algorithms included in our RL-eMPC framework.
- [x] Threshold-based.
- [x] LSTDQ.
- [x] DQN.
- [x] DDQN.
- [x] DQN/DDQN + PER (prioritized experience buffer) + LSTM
- [x] A2C (Discrete)
- [x] PPO (Discrete)
- [x] SAC (Discrete)

To visualize the training process, you can run:  `tensorboard --logdir runs`, for instance:
<img src="https://user-images.githubusercontent.com/25771207/174673227-7f8fbaae-ddcb-437b-bd50-588f2de94ee8.png" width="700" height="500">
<img src="https://user-images.githubusercontent.com/25771207/174673245-efeb39b5-2f6e-4350-be2e-8eecebb5f3fb.png" width="200" height="80">

## References
The A2C, PPO, and SAC code are based on the following wonderful repos, please give the credits to the authors.
- [A2C](https://github.com/dongminlee94/deep_rl)
- [PPO](https://github.com/RPC2/PPO)
- [SAC](https://github.com/ku2482/sac-discrete.pytorch)

## Cite
Please consider cite our paper if you find this repo is useful.
https://urldefense.com/v3/__http://arxiv.org/abs/2208.10302__;!!HXCxUKc!2bA5zV6LbesRBR_dmzZQ_96creDx9NU6EPGTNBMnAGMxbySbWfm86Qz07MCOAnk5yK9hmxiO7Of5UxU9Pg$  
