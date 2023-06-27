import os.path as osp

from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize

def setup_isaac_gym(exp_cfg):
    def omegaconf_to_dict(d):
        ret = {}
        for k, v in d.items():
            if isinstance(v, DictConfig):
                ret[k] = omegaconf_to_dict(v)
            else:
                ret[k] = v
        return ret

    # load isaacgym config
    with initialize(config_path="../config/isaacgym_cfg"):
        cfg = compose(config_name="config", overrides=["task={}".format(exp_cfg.env_name)])
        cfg_dict = omegaconf_to_dict(cfg)
    
    # overwrite numEnvs and render from CLI args
    cfg_dict['headless'] = not exp_cfg.render
    cfg_dict['task']['env']['numEnvs'] = exp_cfg.num_envs

    # overwrite training params
    cfg_dict['train']['params']['config']['num_actors'] = exp_cfg.num_envs
    cfg_dict['num_players'] = exp_cfg.num_players
    cfg_dict['task']['disable_joints'] = exp_cfg.disable_joints

    # load supervisors
    rootdir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    checkpoints = []

    if exp_cfg.num_players == 1:
        checkpoints.append(osp.join(rootdir, 'env/assets/isaacgym/supervisors/{}{}.pth'.format(exp_cfg.env_name, exp_cfg.use_player)))
    else: 
        for player in range(exp_cfg.num_players): # assuming zero-indexed players. 
            checkpoints.append(osp.join(rootdir, 'env/assets/isaacgym/supervisors/{}{}.pth'.format(exp_cfg.env_name, player)))
    
    cfg_dict['checkpoints'] = checkpoints
    
    return cfg_dict