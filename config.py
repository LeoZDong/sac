"""Argument parser for configurations"""

import os
import argparse
import yaml
import io

import tensorflow as tf

from util import str2bool

def get_parser():
    parser = argparse.ArgumentParser(
        description="Arguments for RL training and evaluation"
    )

    # Session configuration
    parser.add_argument('--name', type=str, default='default', 
                        help="Name of this session / model to be tagged.")
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v2',
                        help="Name of environment to train in.")
    parser.add_argument('--use_tf_functions', type=str2bool, default=True, 
                        help="Whether to wrap functions in tf functions.")
    parser.add_argument('--resume', action='store_true', 
                        help="Whether to resume from the last checkpoint.")

    # Training setup
    parser.add_argument('--n_iter', type=int, default=3000000,
                        help='Number of training iterations (epochs).')
    parser.add_argument('--n_steps_per_iter', type=int, default=1, 
                        help="Number of training steps per iteration.")
    parser.add_argument('--batch_size', type=int, default=256, 
                        help="Batch size.")
    parser.add_argument('--n_collect_init', type=int, default=10000,
                        help="Initial collection steps.")
    # optimization hyperparameters
    parser.add_argument('--lr_critic', type=float, default=3e-4,
                        help="Learning rate for critic.")
    parser.add_argument('--lr_actor', type=float, default=3e-4,
                        help="Learning rate for actor.")
    parser.add_argument('--lr_alpha', type=float, default=3e-4,
                        help="Alpha of learning rate")
    parser.add_argument('--tgt_tau', type=float, default=0.005,
                        help="Target update tau?")
    parser.add_argument('--tgt_update_period', type=float, default=1,
                        help="Target update period.")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="Gamma? for SAC agent")
    parser.add_argument('--r_scale', type=float, default=1,
                        help="Reward scale factor.")

    # Data collection setup
    parser.add_argument('--n_collect_per_iter', type=int, default=1,
                        help="Collection steps per iteration.")
    parser.add_argument('--rb_len', type=int, default=1000000,
                        help="Replay buffer maximum length.")

    # Evaluation setup
    parser.add_argument('--n_eval_eps', type=int, default=20, 
                        help="Number of evaluation episodes.")
    parser.add_argument('--eval_interval', type=int, default=10000, 
                        help="Evaluate every n iters.")
    
    # Data loading and model saving
    parser.add_argument('--train_ckpt_interval', type=int, default=50000, 
                        help="Save training checkpoints every n iters.")
    parser.add_argument('--policy_ckpt_interval', type=int, default=50000,
                        help="Save policy checkpoints every n iters.")
    parser.add_argument('--rb_ckpt_interval', type=int, default=50000,
                        help="Save replay buffer checkpoints every n iters.")
    parser.add_argument('--log_interval', type=int, default=1000, 
                        help="Log stats every n iters.")
    parser.add_argument('--summary_interval', type=int, default=1000,
            help="Write summary every n iters for interface with tensorboard.")
    parser.add_argument('--summaries_flush_secs', type=int, default=10,
                        help="Flush summary memory every n seconds.")

    parser.add_argument('--train_dir', type=str, default='train', 
                        help="Directory to save training models and viz.")
    parser.add_argument('--eval_dir', type=str, default='eval', 
                        help="Directory to save evaluation models and viz.")

    return parser


def parse(root=os.path.dirname(os.path.abspath(__file__)), config_file=None, 
          save_config=False):
    parser = get_parser()
    if config_file is not None:
        with open(config_file, 'r') as stream:
            config = yaml.safe_load(stream)
            args = parser.parse_args(config)
    else:
        args = parser.parse_args()

    # Configure directories
    args.train_dir = os.path.join(root, args.train_dir, args.name)
    args.eval_dir = os.path.join(root, args.eval_dir, args.name)
    
    # Create directories if needed
    if not os.path.isdir(args.train_dir):
        os.makedirs(args.train_dir)
    if not os.path.isdir(args.eval_dir):
        os.makedirs(args.eval_dir)
    
    if save_config:
        save_file = os.path.join(args.viz_dir, 
                                 'config_{}.yaml'.format(args.model_name))
        with io.open(save_file, 'w') as outfile:
            yaml.dump(args, outfile)

    return args

def get_optimizer(args):
    return tf.keras.optimizers.Adam(learning_rate=args.lr)
