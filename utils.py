import os
from datetime import datetime
from dotmap import DotMap
import omegaconf
import yaml, json, pickle
import warnings

def datetime_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def print_recursive(obj, indent=0, step=4):
    obj_type = type(obj)

    print(' ' * indent + str(obj_type) + ':', end='')

    if obj_type in [dict, DotMap]:
        print('')
        for key in obj:
            print(' ' * (indent + step) + str(key) + ':')
            print_recursive(obj[key], indent + 2 * step, step)
    elif obj_type in [list, tuple, omegaconf.listconfig.ListConfig]:
        print('')
        for item in obj:
            print_recursive(item, indent + step, step)
    else:
        print(' '+ str(obj))



def get_recursive_types(obj, obj_types=[]):
    obj_type = type(obj)

    new_obj_types = [obj_type]

    if obj_type in [dict, DotMap, omegaconf.listconfig.ListConfig]:
        for key in obj:
            new_obj_types += get_recursive_types(obj[key], obj_types)
    elif obj_type in [list, tuple]:
        for item in obj:
            new_obj_types += get_recursive_types(item, obj_types)
    
    for new_obj_type in new_obj_types:
        if new_obj_type not in obj_types:
            obj_types.append(new_obj_type)
    
    return obj_types


def save_config(cfg, filename):
    extension = os.path.splitext(filename)[1][1:]

    if extension not in ['pickle', 'yaml', 'json']:
        raise ValueError('Invalid file extension: ' + str(extension))

    # Pickle files can be saved without formatting
    if extension == 'pickle':
        with open(filename, 'wb') as out_file:
            pickle.dump(cfg, out_file)
        return
    
    # YAML and JSON files require formatting to avoid invalid datatype errors
    cfg = _json_formatting(cfg)

    with open(filename, 'w') as out_file:
        if extension == 'yaml':
            yaml.dump(cfg, out_file)
        else:
            json.dump(cfg, out_file, indent=4)


def load_config(filename):
    extension = os.path.splitext(filename)[1][1:]

    if extension not in ['pickle', 'yaml', 'json']:
        raise ValueError('Invalid file extension: ' + str(extension))

    if extension == 'pickle':
        with open(filename, 'rb') as in_file:
            cfg = pickle.load(in_file)
        return cfg
    else:
        with open(filename, 'r') as in_file:
            if extension == 'yaml':
                cfg = yaml.load(in_file, Loader=yaml.FullLoader)
            else:
                cfg = json.load(in_file)

    return cfg

def find_experiment_args_filename(logdir):
    for filename in os.listdir(logdir):
        base_name = os.path.basename(filename)
        if os.path.splitext(base_name)[0] == 'args':
            return base_name

    raise FileNotFoundError('Could not find \'args.*\' file in ' + logdir)
    
    
def _json_formatting(obj):
    if type(obj) in [dict, DotMap, omegaconf.dictconfig.DictConfig]:
        new_obj = {}

        for key in obj:
            new_obj[key] = _json_formatting(obj[key])

        return new_obj
    
    if type(obj) in [list, tuple, omegaconf.listconfig.ListConfig]:
        new_obj = []

        for i in range(len(obj)):
            new_obj.append(_json_formatting(obj[i]))
        
        return new_obj

    if type(obj) in [int, float, bool, str]:
        return obj

    return str(obj)

"""A set of common utilities used within the environments.

These are not intended as API functions, and will not remain stable over time.
"""

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)

### Taken from gym/utils/colorize.py
def colorize(
    string: str, color: str, bold: bool = False, highlight: bool = False
) -> str:
    """Returns string surrounded by appropriate terminal colour codes to print colourised text.

    Args:
        string: The message to colourise
        color: Literal values are gray, red, green, yellow, blue, magenta, cyan, white, crimson
        bold: If to bold the string
        highlight: If to highlight the string

    Returns:
        Colourised string
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    attrs = ";".join(attr)
    return f"\x1b[{attrs}m{string}\x1b[0m"

def warn(message):
    warnings.warn(colorize('WARN: ' + message, "yellow"))


def get_pretraining_name(config):
    experiment_name = '{}_seed-{}_{}_{}_{}'.format(
        datetime_str(),
        config.seed,
        config.env_name,
        config.method,
        config.logdir_suffix
    )
    return experiment_name