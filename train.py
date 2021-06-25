from __future__ import absolute_import, division, print_function

import os
import sys
import time
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay
from absl import logging

import tensorflow as tf

from tf_agents.environments import suite_mujoco, suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils

import config
import util
import models
from training import Trainer

def train(args):
    try:
        collect_py_env = suite_mujoco.load(args.env_name)
        eval_py_env = suite_mujoco.load(args.env_name)
    except:
        collect_py_env = suite_gym.load(args.env_name)
        eval_py_env = suite_gym.load(args.env_name)

    collect_py_env.reset()
    collect_env = tf_py_environment.TFPyEnvironment(collect_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    # PIL.Image.fromarray(train_py_env.render())
    # train_py_env.render()

    # Print information about the environment
    print('{} train environment:'.format(args.env_name))
    print('Observation Spec:')
    print(collect_py_env.time_step_spec().observation)
    print('Reward Spec:')
    print(collect_py_env.time_step_spec().reward)
    print('Action Spec:')
    print(collect_py_env.action_spec())

    # Obtain agent and trainer
    # global step counter
    global_step = tf.compat.v1.train.get_or_create_global_step()
    agent = models.get_agent(collect_env, global_step, args)
    trainer = Trainer(agent, collect_env, eval_env, args)

    # Load checkpoint
    if args.resume:
        trainer.checkpointer['train'].initialize_or_restore()
        trainer.checkpointer['rb'].initialize_or_restore()

    # TODO: Evaluate the agent's policy before training
    # avg_return = trainer.get_eval_metrics()['AverageReturn']
    # returns = [avg_return]

    # timed_at_step = global_step.numpy()
    for i in range(args.n_iter):
        train_loss = trainer.train_iter()
        step = global_step.numpy()
        write_summary = lambda: tf.math.equal(
            global_step % args.summary_interval, 0)

        if args.log_interval and step % args.log_interval == 0:
            with tf.compat.v2.summary.record_if(write_summary):
                trainer.log_info()
        
        if args.eval_interval and step % args.eval_interval == 0:
            with tf.compat.v2.summary.record_if(write_summary):
                trainer.eval_iter()
        
        if args.train_ckpt_interval and step % args.train_ckpt_interval == 0:
            trainer.checkpointer['train'].save(global_step=step)
        
        if args.policy_ckpt_interval and step % args.policy_ckpt_interval == 0:
            trainer.checkpointer['policy'].save(global_step=step)
        
        if args.rb_ckpt_interval and step % args.rb_ckpt_interval == 0:
            trainer.checkpointer['rb'].save(global_step=step)
    
    util.create_policy_eval_video(trainer.agent.eval_policy, eval_env, 
                                  eval_py_env, args.viz_dir, 'trained-agent')

def main():
    start_time = time.time()

    tf.compat.v1.enable_v2_behavior()
    # Set up a virtual display for rendering OpenAI gym environments.
    display = pyvirtualdisplay.Display(visible=False, size=(1400, 900)).start()

    if os.getcwd().startswith('/sailhome'):
        root = '/iris/u/leozdong/rl_playground'
    else:
        root = os.getcwd()
    arguments = config.parse(root=root)

    # Configure logger
    logging.get_absl_handler().use_absl_log_file(
        '{}_{}.log'.format(arguments.name, arguments.env_name), './')
    logging.set_verbosity(logging.INFO)

    args_info = ['===== Arguments BEGIN =====']
    for key, val in vars(arguments).items():
        args_info.append('{:20} {}'.format(key, val))
    args_info.append('====== Arguments END ======')
    args_info = '\n'.join(args_info)

    print(args_info)
    logging.info(args_info)

    print("Initialization time: {:.3f}".format(time.time() - start_time))

    train(arguments)

if __name__  == '__main__':
    main()

    
    

