import os
import gym
import time
import argparse
import datetime
import numpy as np
import torch
import csv
import random
from torch.utils.tensorboard import SummaryWriter
from veh_env2 import vehEnv
from scipy.io import savemat

seed = 0    #####positive integer
random.seed(seed)
# Set a random seed
np.random.seed(seed)
torch.manual_seed(seed)

# Configurations
parser = argparse.ArgumentParser(description='RL algorithms with PyTorch in CartPole environment')
parser.add_argument('--algo', type=str, default='a2c',
                    help='select an algorithm among dqn, ddqn, a2c')
parser.add_argument('--phase', type=str, default='train',
                    help='choose between training phase and testing phase')
parser.add_argument('--render', action='store_true', default=False,
                    help='if you want to render, set this to True')
parser.add_argument('--load', type=str, default=None,
                    help='copy & paste the saved model name, and load it')
parser.add_argument('--seed', type=int, default=0, 
                    help='seed for random number generators')
parser.add_argument('--iterations', type=int, default=1000,
                    help='iterations to run and train agent')
parser.add_argument('--eval_per_train', type=int, default=5,
                    help='evaluation number per training')
parser.add_argument('--max_step', type=int, default=100,
                    help='max episode step')
parser.add_argument('--tensorboard', action='store_true', default=True)
parser.add_argument('--gpu_index', type=int, default=0)
args = parser.parse_args()
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

if args.algo == 'dqn':
    from agents.dqn import Agent
elif args.algo == 'ddqn': # Just replace the target of DQN with Double DQN
    from agents.dqn import Agent
elif args.algo == 'a2c':
    from agents.a2c import Agent


def main():
    """Main."""
    # Initialize environment
    rho_c = 0.01
    env = vehEnv(T=20, rho=rho_c)
    obs_dim = env.obs_space
    act_num = env.action_space
    
    print('---------------------------------------')
    print('Algorithm:', args.algo)
    print('State dimension:', obs_dim)
    print('Action number:', act_num)
    print('---------------------------------------')

    # Create an agent
    agent = Agent(env, args, device, obs_dim, act_num)

    # If we have a saved model, load it
    if args.load is not None:
        pretrained_model_path = os.path.join('./save_model/' + str(args.load))
        pretrained_model = torch.load(pretrained_model_path, map_location=device)
        if args.algo == 'dqn' or args.algo == 'ddqn':
            agent.qf.load_state_dict(pretrained_model)
        else:
            agent.policy.load_state_dict(pretrained_model)

    # Create a SummaryWriter object by TensorBoard
    if args.tensorboard and args.load is None:
        dir_name = 'runs' + '/' \
                           + args.algo \
                           + '_rho_' + str(rho_c) \
                           + '_t_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        writer = SummaryWriter(log_dir=dir_name)

    if not os.path.isfile(dir_name + '/train_logs.csv'):
        with open(dir_name + '/train_logs.csv', mode='w') as csv_file:
            fieldnames = ['Model', 'Epochs', 'max_step', 'Training Return', 'Evaluation Return', "Evaluation Jmpc",
                          'Mean Error', 'Min_from_15t', 'Max_from_15t', 'action_freq']
            writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer_csv.writeheader()

    start_time = time.time()

    train_num_steps = 0
    train_sum_returns = 0.
    train_num_episodes = 0

    # Main loop
    for i in range(args.iterations):
        # Perform the training phase, during which the agent learns
        if args.phase == 'train':
            agent.eval_mode = False
        
            # Run one episode
            train_step_length, train_episode_return, action_ratio = agent.run(args.max_step)
            
            train_num_steps += train_step_length
            train_sum_returns += train_episode_return
            train_num_episodes += 1

            # Log experiment result for training episodes
            if args.tensorboard and args.load is None:
                writer.add_scalar('Train/steps', train_step_length, i)
                writer.add_scalar('Train/EpisodeReturns', train_episode_return, i)
                writer.add_scalar('Train/action_ratio', action_ratio, i)

        # Perform the evaluation phase -- no learning
        if (i + 1) % args.eval_per_train == 0:
            agent.eval_mode = True
            
            # Run one episode
            eval_step_length, eval_episode_return, (act, x, rl, gt, jmpcs) = agent.test(args.max_step)
            action_ratio = sum(act) / len(act)

            # Log experiment result for evaluation episodes
            if args.tensorboard and args.load is None:
                writer.add_scalar('Eval/steps', eval_step_length, i + 1)
                writer.add_scalar('Eval/EpisodeReturns', eval_episode_return, i + 1)
                writer.add_scalar('Eval/action_ratio', action_ratio, i + 1)

            save_mat = {
                'act': act,
                'x': x,
                'rl': rl,
                'gt': gt,
                'jmpcs': jmpcs
            }
            if not os.path.exists(dir_name + '/results'):
                os.mkdir(dir_name + '/results')

            savemat(dir_name + '/results/a2c_results_{}.mat'.format(i + 1), save_mat)

            if args.phase == 'train':
                print('---------------------------------------')
                print('Iterations:', i + 1)
                print('Steps:', train_num_steps)
                print('Episodes:', train_num_episodes)
                print('EpisodeReturn:', round(train_episode_return, 2))
                print('EvalEpisodeReturn:', round(eval_episode_return, 2))
                print('OtherLogs:', agent.logger)
                print('Time:', int(time.time() - start_time))
                print('---------------------------------------')

                # Save the trained model
                if not os.path.exists(dir_name + '/save_model'):
                    os.mkdir(dir_name + '/save_model')

                ckpt_path = os.path.join(dir_name + '/save_model/' + '_' + args.algo \
                                                                    + '_s_' + str(args.seed) \
                                                                    + '_i_' + str(i + 1) + '.pt')
                    
                if args.algo == 'dqn' or args.algo == 'ddqn':
                    torch.save(agent.qf.state_dict(), ckpt_path)
                else:
                    torch.save(agent.policy.state_dict(), ckpt_path)
            elif args.phase == 'test':
                print('---------------------------------------')
                print('EvalEpisodeReturn:', round(eval_episode_return, 2))
                print('---------------------------------------')

            max_step = len(rl)
            eval_error = np.array(rl[:max_step]) - 4 * np.sin(2 * np.pi / 50 * np.array(x[:max_step]))

            with open(dir_name + '/train_logs.csv', 'a+', newline='') as write_obj:
                csv_writer = csv.writer(write_obj)
                try:
                    csv_writer.writerow(['A2C', i + 1, max_step, '{:.3f}'.format(train_episode_return),
                                         '{:.3f}'.format(eval_episode_return), '{:.3f}'.format(sum(jmpcs)),
                                         '{:.3f}'.format(np.mean(np.absolute(eval_error))),
                                         '{:.3f}'.format(np.min(eval_error[15:])),
                                         '{:.3f}'.format(np.max(eval_error[15:])),
                                         '{:.3f}'.format(action_ratio)])
                except:
                    csv_writer.writerow(
                        ['A2C', i + 1, max_step, '{:.3f}'.format(train_episode_return),
                         '{:.3f}'.format(eval_episode_return), '{:.3f}'.format(sum(jmpcs)),
                         '{:.3f}'.format(np.mean(np.absolute(eval_error))),
                         '{:.3f}'.format(np.min(eval_error)), '{:.3f}'.format(np.max(eval_error)),
                         '{:.3f}'.format(action_ratio)])


if __name__ == "__main__":
    main()
