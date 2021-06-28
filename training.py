"""Trainer for model."""

import os
import time
from absl import logging

import tensorflow as tf
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_mujoco
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common


class Trainer(object):
    """Trainer class."""
    def __init__(self, agent, collect_env, eval_env, args):
        #### Training configurations ####
        self.agent = agent
        self.collect_env = collect_env
        self.eval_env = eval_env
        self.global_step = self.agent.train_step_counter
        if args.use_tf_functions:
            self.agent.train = common.function(self.agent.train)
            self.train_step = common.function(self.train_step)
        self.n_steps_per_iter = args.n_steps_per_iter
        
        # Extract policies from agent
        self.collect_policy = self.agent.collect_policy
        self.eval_policy = greedy_policy.GreedyPolicy(self.agent.policy)

        # Define train and eval metrics
        metrics = self.build_metrics(args)
        self.train_metrics = metrics['train']
        self.eval_metrics = metrics['eval']

        #### Data collection ####
        # Set initial policy state and time step for collect drivers to run
        self.time_step = None
        self.policy_state = self.collect_policy.get_initial_state(
            collect_env.batch_size)

        # Get replay buffer and dataset iterator
        self.replay_buffer, self.iterator = self.build_rb(args)

        # Get data collection driver
        self.collect_driver = self.build_collect_driver(args)

        # Initial data collection
        self.start_init_collect(args)
        
        #### Evaluation and checkpoints ####
        self.n_eval_eps = args.n_eval_eps
        self.checkpointer = self.build_checkpointers(args)

        ### Logging, visualization, and model saving ####
        self.train_summary_writer = tf.compat.v2.summary.create_file_writer(
            args.train_dir, flush_millis=args.summaries_flush_secs * 1000)
        self.eval_summary_writer = tf.compat.v2.summary.create_file_writer(
            args.eval_dir, flush_millis=args.summaries_flush_secs * 1000)
        
        self.name = args.name


    def train_iter(self):
        """One iteration of training. May contain multiple train steps."""
        start_time = time.time()

        # Collect data
        t = time.time()
        self.collect_step()
        for i in range(self.n_steps_per_iter):
            t = time.time()
            train_loss = self.train_step()
        
        # Gradient update
        experience, _ = next(self.iterator)
        train_loss = self.agent.train(experience)

        # Save records
        self.last_train_loss = train_loss.loss
        self.last_train_time = time.time() - start_time

        return train_loss
    
    def train_step(self):
        """One step of training."""
        experience, _ = next(self.iterator)
        return self.agent.train(experience)
    
    def collect_step(self):
        self.time_step, self.policy_state = self.collect_driver.run(
            self.time_step, self.policy_state)

    def eval_iter(self):
        """One iteration of evaluation."""
        eval_results = self.get_eval_results()

        # Manually perform metric_utils.log_metrics(eval_results)
        # the tf agent function does not work
        log = []
        for metric_name, metric_val in eval_results.items():
            logging.info('{} = {}'.format(metric_name, metric_val.numpy()))
        
        # logging.info('%s \n\t\t %s', '', '\n\t\t'.join(log))
    
    def get_eval_results(self):
        """Evaluate the pre-defined set of evaluation metrics."""
        eval_results = metric_utils.eager_compute(
            self.eval_metrics,
            self.eval_env,
            self.eval_policy,
            num_episodes=self.n_eval_eps,
            train_step=self.global_step,
            summary_writer=self.eval_summary_writer,
            summary_prefix='Metrics'
        )

        return eval_results

    def log_info(self):
        """Log the training information."""
        logging.info("step = {}, loss = {:.3f}, {:.3f} secs/step".format(
            self.global_step.numpy(), 
            self.last_train_loss, 
            self.last_train_time))
        # logging.info("{:.3f} secs/step".format(self.last_train_time))
        
        with self.train_summary_writer.as_default():
            tf.compat.v2.summary.scalar(
                name='secs_per_step', 
                data=self.last_train_time, 
                step=self.global_step)

    
    def step_counter(self):
        """Return the value of current train step counter."""
        return self.agent.train_step_counter.numpy()
    
    def build_checkpointers(self, args):
        """Create instances of relevant checkpointers."""
        train_checkpointer = common.Checkpointer(
            ckpt_dir=args.train_dir,
            agent=self.agent,
            global_step=self.global_step,
            metrics=metric_utils.MetricsGroup(self.train_metrics, 
                'train_metrics'))

        policy_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(args.train_dir, 'policy'),
            policy=self.eval_policy,
            global_step=self.global_step)

        rb_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(args.train_dir, 'replay_buffer'),
            max_to_keep=1,
            replay_buffer=self.replay_buffer)

        return {'train': train_checkpointer, 'policy': policy_checkpointer, 
                'rb': rb_checkpointer}
    
    def build_metrics(self, args):
        """Build instances of relevant metrics."""
        train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(
                buffer_size=args.n_eval_eps, 
                batch_size=self.collect_env.batch_size),
            tf_metrics.AverageEpisodeLengthMetric(
                buffer_size=args.n_eval_eps, 
                batch_size=self.collect_env.batch_size),
        ]

        eval_metrics = [
            tf_metrics.AverageReturnMetric(buffer_size=args.n_eval_eps),
            tf_metrics.AverageEpisodeLengthMetric(
                buffer_size=args.n_eval_eps)
        ]

        return {'train': train_metrics, 'eval': eval_metrics}
    
    def build_rb(self, args):
        """Build replay buffer and dataset iterator."""
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=1,
            max_length=args.rb_len)

        # Prepare replay buffer as dataset with invalid transitions filtered.
        def _filter_invalid_transition(trajectories, *args):
            return ~trajectories.is_boundary()[0]
        dataset = replay_buffer.as_dataset(
            sample_batch_size=args.batch_size,
            num_steps=2).unbatch().filter(
                _filter_invalid_transition).batch(args.batch_size).prefetch(5)

        # Dataset generates trajectories with shape [B x 2 x ...]
        iterator = iter(dataset)

        return replay_buffer, iterator
    
    def build_collect_driver(self, args):
        """Build driver class for data collection."""
        replay_observer = [self.replay_buffer.add_batch]

        collect_driver = dynamic_step_driver.DynamicStepDriver(
            self.collect_env,
            self.collect_policy,
            observers=replay_observer + self.train_metrics,
            num_steps=args.n_collect_per_iter)
        
        return collect_driver

    def start_init_collect(self, args):
        """Start the initial collection process."""
        if self.replay_buffer.num_frames() > 0:
            logging.info(
                ("Replay buffer already stores data." 
                 "Skip initial collection").format(args.n_collect_init))
            return

        initial_collect_policy = random_tf_policy.RandomTFPolicy(
            self.collect_env.time_step_spec(), self.collect_env.action_spec())
        
        # Configure driver
        replay_observer = [self.replay_buffer.add_batch]
        initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
            self.collect_env,
            initial_collect_policy,
            observers=replay_observer + self.train_metrics,
            num_steps=args.n_collect_init)

        if args.use_tf_functions:
            initial_collect_driver.run = common.function(
                initial_collect_driver.run)

        # Collect initial replay data
        logging.info(
            ("Initializing replay buffer by collecting experience for {} steps "
             "with a random policy.").format(args.n_collect_init))
        initial_collect_driver.run()
