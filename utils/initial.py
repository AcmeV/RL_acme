import json
import os

import env

def initial_env(args):
    env_config_file = open(f'{args.config_dir}/env_config.json', 'r')
    config_dict = json.load(env_config_file)

    return eval(f'{config_dict[args.env_type][args.env_name]}()')
