"""Utility functions for the model."""

import base64
import IPython
import imageio
import os
import re
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# def embed_mp4(filename):
#     """Embeds an mp4 file in the notebook."""
#     video = open(filename, 'rb').read()
#     b64 = base64.b64encode(video)
#     tag = '''
#     <video width="640" height="480" controls>
#     <source src="data:video/mp4;base64,{0}" type="video/mp4">
#     Your browser does not support the video tag.
#     </video>'''.format(b64.decode())

#     return IPython.display.HTML(tag)


def create_policy_eval_video(policy, env, py_env, filename, save_dir, 
                             num_episodes=1, fps=30):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    filename = filename + ".mp4"
    filepath = os.path.join(save_dir, filename)
    with imageio.get_writer(filepath, fps=fps) as video:
        for _ in range(num_episodes):
            time_step = env.reset()
            video.append_data(py_env.render())
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = env.step(action_step.action)
                video.append_data(py_env.render())
    # return embed_mp4(filename)

def purge_logs(log_dir):
    # dir = os.path.join(root, )
    # pattern = '^.*INFO.*$'
    pattern = '^.*log'
    for f in os.listdir(log_dir):
        if re.search(pattern, f):
            os.remove(os.path.join(log_dir, f))

def purge_summaries(train_dir, eval_dir):
    dirs = [train_dir, eval_dir]
    pattern = 'events*'
    for root_dir in dirs:
        for f in os.listdir(root_dir):
            if re.search(pattern, f):
                os.remove(os.path.join(root_dir, f))

def purge(args):
    purge_logs(args.log_dir)
    purge_summaries(args.train_dir, args.eval_dir)
