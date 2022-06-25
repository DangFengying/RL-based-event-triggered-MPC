import os
import math, random
import csv
import numpy as np
import yaml
import argparse
from datetime import datetime
import torch

from veh_env2 import vehEnv
from sacd.agent import SacdAgent, SharedSacdAgent

USE_CUDA = False
# Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs)
################################tune parameter:seed##############################
seed = 0    #####positive integer
random.seed(seed)
# Set a random seed
np.random.seed(seed)
torch.manual_seed(seed)


def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    # parameter
    rho_c = 0
    env = vehEnv(T=20, rho=rho_c)
    test_env = vehEnv(T=20, rho=rho_c)

    # Specify the directory to log.
    name = args.config.split('/')[-1].rstrip('.yaml')
    if args.shared:
        name = 'shared-' + name
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'runs', f'{name}-{time}-{rho_c}')

    # Create the agent.
    Agent = SacdAgent if not args.shared else SharedSacdAgent
    agent = Agent(
        env=env, test_env=test_env, log_dir=log_dir, cuda=args.cuda,
        seed=args.seed, **config)
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default='sacd.yaml')
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run(args)
