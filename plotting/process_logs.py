import argparse
import os
import pickle
import numpy as np
from dotmap import DotMap

from utils import find_experiment_args_filename, load_config


def process_stats(exp_cfg, raw_data, envs, exp_stats=None):
    curr_timesteps = len(raw_data)

    if type(exp_cfg) == dict:
        exp_cfg = DotMap(exp_cfg)

    if exp_stats == None:
        exp_stats = init_empty_stats(exp_cfg)
    
    # Calculate missing data
    past_timesteps = len(exp_stats['cumulative_viols']) - 1

    if past_timesteps == 0:
        assignments = np.zeros((exp_cfg.num_envs, exp_cfg.num_humans))
    else:
        assignments = np.copy(raw_data[past_timesteps-1]['assignments'])

    for t in range(past_timesteps, curr_timesteps):
        num_switches = np.sum(np.abs(assignments - raw_data[t]['assignments']).sum(1) > 0)
        exp_stats['total_switches'] += num_switches
        exp_stats['cumulative_switches'].append(exp_stats['cumulative_switches'][-1] + num_switches)
        assignments = np.copy(raw_data[t]['assignments'])
        exp_stats['cumulative_viols'].append(exp_stats['cumulative_viols'][-1])
        exp_stats['cumulative_successes'].append(exp_stats['cumulative_successes'][-1])
        exp_stats['cumulative_hard_resets'].append(exp_stats['cumulative_hard_resets'][-1])
        exp_stats['cumulative_human_actions'].append(exp_stats['cumulative_human_actions'][-1])
        exp_stats['cumulative_idle_time'].append(exp_stats['cumulative_idle_time'][-1])
        exp_stats['cumulative_reward'].append(exp_stats['cumulative_reward'][-1])
        
        for i in range(exp_cfg.num_envs):
            exp_stats[i]['cumulative_hard_resets'].append(exp_stats[i]['cumulative_hard_resets'][-1])
            exp_stats[i]['cumulative_human_actions'].append(exp_stats[i]['cumulative_human_actions'][-1])
            exp_stats[i]['cumulative_idle_time'].append(exp_stats[i]['cumulative_idle_time'][-1])
            exp_stats[i]['cumulative_viols'].append(exp_stats[i]['cumulative_viols'][-1])
            exp_stats[i]['cumulative_successes'].append(exp_stats[i]['cumulative_successes'][-1])
            use_human_ac = np.sum(assignments[i]) == 1
            if raw_data[t]['reward'][i]:
                exp_stats['total_reward'] += raw_data[t]['reward'][i]
                exp_stats['cumulative_reward'][-1] += raw_data[t]['reward'][i]
            if use_human_ac and raw_data[t]['constraint'][i]:
                exp_stats[i]['num_hard_resets'] += 1
                exp_stats[i]['cumulative_hard_resets'][-1] += 1
                exp_stats['total_hard_resets'] += 1
                exp_stats['cumulative_hard_resets'][-1] += 1
            if use_human_ac:
                exp_stats['total_human_actions'] += 1
                exp_stats['cumulative_human_actions'][-1] += 1
                exp_stats[i]['num_human_actions'] += 1
                exp_stats[i]['cumulative_human_actions'][-1] += 1
            if raw_data[t]['idle'][i]:
                exp_stats[i]['idle_time'] += 1
                exp_stats[i]['cumulative_idle_time'][-1] += 1
                exp_stats['total_idle_time'] += 1
                exp_stats['cumulative_idle_time'][-1] += 1
            if raw_data[t]['info'][i] and raw_data[t]['info'][i]['constraint']:
                exp_stats[i]['num_viols'] += 1
                exp_stats['total_viols'] += 1
                exp_stats[i]['cumulative_viols'][-1] += 1
                exp_stats['cumulative_viols'][-1] += 1
            if raw_data[t]['info'][i] and raw_data[t]['info'][i]['success']:
                exp_stats[i]['num_successes'] += 1
                exp_stats['total_successes'] += 1
                exp_stats[i]['cumulative_successes'][-1] += 1
                exp_stats['cumulative_successes'][-1] += 1
    
    max_steps = envs.max_episode_steps if exp_cfg.vec_env else envs[0].max_episode_steps
    exp_stats['average_reward'] = \
        exp_stats['total_reward'] \
        * max_steps \
        / curr_timesteps \
        / exp_cfg.num_envs

    exp_stats['ROHE'] = exp_stats['total_reward'] / (exp_stats['total_human_actions'] + 1)
    
    return exp_stats


def compute_stats(logdir, envs, exp_stats=None):
    # Load raw data
    raw_data = pickle.load(open(os.path.join(logdir, 'raw_data.pkl'), 'rb'))

    args_filename = find_experiment_args_filename(logdir)
    exp_cfg = load_config(os.path.join(logdir, args_filename))


    # Process data
    exp_stats = process_stats(exp_cfg, raw_data, envs, exp_stats)


    # Save and return processed data
    pickle.dump(exp_stats, open(os.path.join(logdir, 'run_stats.pkl'), 'wb'))

    return len(raw_data), exp_stats['total_reward'], exp_stats['total_successes'], exp_stats['total_viols'], \
        exp_stats['total_switches'], exp_stats['total_human_actions'], exp_stats['total_idle_time']


def init_empty_stats(exp_cfg):
    exp_stats = {
        'total_successes': 0,
        'total_viols': 0,
        'total_hard_resets': 0,
        'total_switches': 0,
        'total_human_actions': 0,
        'total_idle_time': 0,
        'total_reward': 0,
        'cumulative_successes': [0],
        'cumulative_viols': [0],
        'cumulative_hard_resets': [0],
        'cumulative_switches': [0],
        'cumulative_human_actions': [0],
        'cumulative_idle_time': [0],
        'cumulative_reward': [0]
    }
    for i in range(exp_cfg.num_envs):
        exp_stats[i] = {
            'num_successes': 0,
            'num_viols': 0,
            'num_hard_resets': 0,
            'num_human_actions': 0,
            'idle_time': 0,
            'cumulative_successes': [0],
            'cumulative_viols': [0],
            'cumulative_hard_resets': [0],
            'cumulative_human_actions': [0],
            'cumulative_idle_time': [0]
        }
    
    return exp_stats


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('logdir', type=str, help='filepath to log directory (parent of raw_data.pkl)')
#     args = parser.parse_args()
#     result = compute_stats(args.logdir)
#     t, rew, succ, viol, switch, human, idle = compute_stats(args.logdir)
#     print("Steps: %d Successes: %d Violations: %d Switches: %d Human Acts: %d Idle Time: %d"%(
#         t, succ, viol, switch, human, idle))
