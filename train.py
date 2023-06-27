import os.path as osp

from dotmap import DotMap
import wandb
import isaacgym # need this import before other packages to avoid error

from arg_utils import get_parser, add_config_args
from env.utils import setup_isaac_gym
from iflb.parallel_experiment import ParallelExperiment
from iflb.agents import *
from iflb.supervisors import *
from iflb.allocations import *
from utils import warn

import warnings
warnings.filterwarnings("ignore")

WANDB_PROJECT_NAME = None
WANDB_ENTITY_NAME = None

if __name__ == '__main__':
    # Get raw user arguments and construct config
    parser = get_parser()
    exp_cfg, _ = parser.parse_known_args()

    print("CONFIG:", exp_cfg)

    # Create experiment and run it
    # load agent
    rootdir = osp.dirname(osp.abspath(__file__))

    agent = agent_map[exp_cfg.agent]
    filepath = osp.join(rootdir, 'iflb/agents/cfg/{}'.format(agent_cfg_map.get(exp_cfg.agent, 'base_agent.yaml')))
    parser, agent_cfg = add_config_args(parser, filepath)
    
    # load supervisor
    supervisor = supervisor_map[exp_cfg.supervisor]
    filepath = osp.join(rootdir, 'iflb/supervisors/cfg/{}'.format(supervisor_cfg_map.get(exp_cfg.supervisor, 'base_supervisor.yaml')))
    parser, supervisor_cfg = add_config_args(parser, filepath)
    
    # load allocation
    allocation = allocation_map[exp_cfg.allocation]
    filepath = osp.join(rootdir, 'iflb/allocations/cfg/{}'.format(allocation_cfg_map.get(exp_cfg.allocation, 'base_allocation.yaml')))
    parser, allocation_cfg = add_config_args(parser, filepath)

    # Get all arguments
    exp_cfg, unknown_arguments = parser.parse_known_args()

    if len(unknown_arguments) > 0:
        warn('Couldn\'t parse unknown arguments: {}'.format(unknown_arguments))

    exp_cfg = vars(exp_cfg)
    exp_cfg = DotMap(exp_cfg)
    exp_cfg.agent_cfg = DotMap()
    exp_cfg.supervisor_cfg = DotMap()
    exp_cfg.allocation_cfg = DotMap()

    # **NOTE**: Assumes that the keys don't overlap among cfg files!
    for key in agent_cfg:
        exp_cfg.agent_cfg[key] = exp_cfg[key]
        del exp_cfg[key]
    for key in supervisor_cfg:
        exp_cfg.supervisor_cfg[key] = exp_cfg[key]
        del exp_cfg[key]
    for key in allocation_cfg:
        exp_cfg.allocation_cfg[key] = exp_cfg[key]
        del exp_cfg[key]

    if exp_cfg.vec_env:
        # isaac gym conf loading
        ig_cfg = setup_isaac_gym(exp_cfg)
        exp_cfg.isaacgym_cfg = DotMap(ig_cfg)

    print(exp_cfg.agent_cfg)
    wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME if WANDB_PROJECT_NAME != None else 'iifl',
        config=exp_cfg.toDict(),
    )
    
    experiment = ParallelExperiment(exp_cfg, agent, supervisor, allocation)
    experiment.run()
