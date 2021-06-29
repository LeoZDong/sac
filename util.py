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

def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename, 'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
    <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>'''.format(b64.decode())

    return IPython.display.HTML(tag)


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
    return embed_mp4(filename)

def purge_logs(root='./'):
    dir = root
    pattern = '^.*INFO.*$'
    for f in os.listdir(dir):
        print('logs', f)
        if re.search(pattern, f):
            print('match', f)
            os.remove(os.path.join(dir, f))

def purge_summaries(root='./'):
    dirs = [os.path.join(root, 'train'), os.path.join(root, 'eval')]
    pattern = 'events*'
    for dir in dirs:
        for sub_dir in os.listdir(dir):
            for f in os.listdir(os.path.join(dir, sub_dir)):
                print('sum', f)
                if re.search(pattern, f):
                    os.remove(os.path.join(dir, sub_dir, f))

def purge(root='./'):
    purge_logs(root)
    purge_summaries(root)
