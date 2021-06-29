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
import logging

import pyglet
import tensorflow as tf

from tf_agents.environments import suite_mujoco, suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils

import config
import util
import models
from training import Trainer

def train(args):
    if args.purge:
        print("Purging logs and summaries...")
        util.purge()

    try:
        collect_py_env = suite_mujoco.load(args.env_name)
        eval_py_env = suite_mujoco.load(args.env_name)
    except:
        collect_py_env = suite_gym.load(args.env_name)
        eval_py_env = suite_gym.load(args.env_name)

    collect_py_env.reset()
    collect_env = tf_py_environment.TFPyEnvironment(collect_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    import ipdb; ipdb.set_trace()
    print("Test rendering...")
    # GlfwContext(offscreen=True)
    PIL.Image.fromarray(eval_py_env.render())

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

    # Initial evaluation
    start_step = global_step.numpy()
    trainer.eval_iter()

    import ipdb; ipdb.set_trace()
    if args.policy_vid_interval:
        util.create_policy_eval_video(
            trainer.eval_policy, eval_env, eval_py_env,
            os.path.join(args.eval_dir, 'videos'),
            'trained-agent-step{:06d}'.format(global_step.numpy())
        )

    for i in range(start_step, args.n_iter):
        trainer.train_iter()
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

        if args.policy_vid_interval and step % args.policy_vid_interval == 0:
            util.create_policy_eval_video(
                trainer.eval_policy, eval_env, eval_py_env, 
                os.path.join(args.eval_dir, 'videos'),
                'trained-agent-step{:06d}'.format(step)
            )

    if args.policy_vid_interval:
        util.create_policy_eval_video(
            trainer.eval_policy, eval_env, eval_py_env,
            os.path.join(args.eval_dir, 'videos'),
            'trained-agent-step{:06d}-final'.format(step)
        )

def main():
    tf.compat.v1.enable_v2_behavior()
    # Set up a virtual display for rendering OpenAI gym environments.
    display = pyvirtualdisplay.Display(visible=False, size=(1400, 900)).start()

    root = os.getcwd()
    arguments = config.parse(root=root)

    # Configure logger
    log_file = os.path.join(arguments.log_dir, '{}_{}.log'.format(
        arguments.name, arguments.env_name))
    
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M:%S', filemode='w')

    logging.getLogger().addHandler(logging.StreamHandler())

    args_info = ['===== Arguments BEGIN =====']
    for key, val in vars(arguments).items():
        args_info.append('{:20} {}'.format(key, val))
    args_info.append('====== Arguments END ======')
    args_info = '\n'.join(args_info)

    print(args_info)
    logging.info(args_info)

    train(arguments)

if __name__  == '__main__':
    main()

    
    

