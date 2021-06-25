"""Model definitions"""

import tensorflow as tf
# from tensorflow.keras import Model
from tf_agents.train.utils import spec_utils

# SAC
from tf_agents.agents.ddpg import critic_network
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.agents.sac import sac_agent
from tf_agents.train.utils import train_utils


def get_critic_network(train_env, fc_layer_params=(256, 256)):
    """Critic network to give estimates of Q(s, a)."""
    observation_spec, action_spec, time_step_spec = (
        spec_utils.get_tensor_specs(train_env))
    
    # Input: state and action
    # Output: Q value
    return critic_network.CriticNetwork(
        (observation_spec, action_spec),
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=fc_layer_params,
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform')


def get_actor_network(train_env, fc_layer_params=(256, 256)):
    """Actor network to predict parameters for a normal distribution, from which
    we sample actions.
    """
    observation_spec, action_spec, time_step_spec = (
        spec_utils.get_tensor_specs(train_env))

    # Input: state
    # Output: normal distribution over actions
    return actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        fc_layer_params=fc_layer_params,
        continuous_projection_net=(
            tanh_normal_projection_network.TanhNormalProjectionNetwork))


def get_agent(collect_env, global_step, args):
    """SAC agent. """
    # Setup and get relevant networks
    observation_spec, action_spec, time_step_spec = (
        spec_utils.get_tensor_specs(collect_env))

    actor_net = get_actor_network(collect_env)
    critic_net = get_critic_network(collect_env)

    # Create agent
    tf_agent = sac_agent.SacAgent(
            time_step_spec,
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=args.lr_actor),
            critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=args.lr_critic),
            alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=args.lr_alpha),
            target_update_tau=args.tgt_tau,
            target_update_period=args.tgt_update_period,
            td_errors_loss_fn=tf.math.squared_difference,
            gamma=args.gamma,
            reward_scale_factor=args.r_scale,
            train_step_counter=global_step)

    tf_agent.initialize()

    return tf_agent
