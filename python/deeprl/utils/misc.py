#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
# import pickle
# import os
import datetime
# import torch
import time
from pathlib import Path
import itertools
# from collections import OrderedDict, Sequence   ## api changed, by nov05
from collections import OrderedDict               ## api changed, by nov05
from collections.abc import Sequence              ## api changed, by nov05
from tqdm import tqdm
from shutil import rmtree
import json

## local imports
from .torch_utils import *
from ..component.envs import get_env_type


def run_steps(agent):

    config = agent.config
    ## log config 
    log_info = f"\n{config.__repr__()}\n"
    agent.logger.info(log_info)

    agent_name = agent.__class__.__name__
    t0 = time.time()

    for _ in tqdm(range(config.max_steps), desc='Max steps', position=0, leave=True): 
        ## save trained model at intervals
        if config.save_interval and (not (agent.total_steps+1) % config.save_interval):
            agent.save('data/%s-%s-%d' % (agent_name, config.tag, agent.total_steps))
            log_info = f"Model saved as {'data/%s-%s-%d' % (agent_name, config.tag, agent.total_steps)}"
            agent.logger.info(log_info)
        ## log steps/s at intervals
        if config.log_interval and (not (agent.total_steps+1) % config.log_interval):
            time_interval = time.time() - t0
            if time_interval==0:
                log_info = f"steps {agent.total_steps}, - steps/s (time interval = 0)"
            else:
                log_info = 'steps %d, %.2f steps/s' % (agent.total_steps, config.log_interval / time_interval)
            agent.logger.info(log_info)
            t0 = time.time()
        ## log eval result at intervals
        if config.eval_interval and (not (agent.total_steps+1) % config.eval_interval):
            agent.eval_episodes()

        agent.step()
        agent.switch_task()

    agent.close()
    if config.eval_env is not None:
        config.eval_env.close()



def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


def get_default_log_dir(name):
    return './log/%s-%s' % (name, get_time_str())


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def rmdir(path): ## added by nov05
    rmtree(path)
    
    
def close_obj(obj):
    if hasattr(obj, 'close'):
        obj.close()


def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]


def is_plain_type(x):
    for t in [str, int, float, bool]:
        if isinstance(x, t):
            return True
    return False


def generate_tag(params):
    if 'tag' in params.keys():
        return
    params.setdefault('run', 0)
    exclude = ['game', 'run', 'env_fn', 'env_fn_kwargs']
    str = ['%s_%s' % (k, v if is_plain_type(v) else v.__name__) 
           for k, v in sorted(params.items()) if k not in exclude]
    tag = '%s-%s-run-%d' % (params['game'], '-'.join(str), params['run'])
    params['tag'] = tag
    ## e.g. tag is "Reacher-v2-remark_ddpg_continuous-run-0"


def translate(pattern):
    groups = pattern.split('.')
    pattern = (r'\.').join(groups)
    return pattern


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


class HyperParameter:
    def __init__(self, id, param):
        self.id = id
        self.param = dict()
        for key, item in param:
            self.param[key] = item

    def __str__(self):
        return str(self.id)

    def dict(self):
        return self.param


class HyperParameters(Sequence):
    def __init__(self, ordered_params):
        if not isinstance(ordered_params, OrderedDict):
            raise NotImplementedError
        params = []
        for key in ordered_params.keys():
            param = [[key, iterm] for iterm in ordered_params[key]]
            params.append(param)
        self.params = list(itertools.product(*params))

    def __getitem__(self, index):
        return HyperParameter(index, self.params[index])

    def __len__(self):
        return len(self.params)