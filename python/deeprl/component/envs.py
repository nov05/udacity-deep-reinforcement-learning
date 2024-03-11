#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
## changed by nov05, 20240304

# import os
import gym
import numpy as np
# import torch
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.atari_wrappers import FrameStack as FrameStack_
from baselines.common.vec_env import VecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

## local imports
from ..utils import *

import platform
if platform.system()!='Windows':
    ## windows will cause error when importing roboschool  
    ## gym.error.Error: Cannot re-register id: RoboschoolInvertedPendulum-v1
    try:
        import roboschool
        print("roboschool", roboschool.__version__)
    except ImportError as e:
        print(e)
        pass

import sys
import warnings
if not sys.warnoptions:  # allow overriding with `-W` option
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='runpy')
gym.logger.set_level(40)   


## added by nov05
import dm_control2gym
from unityagents import UnityEnvironment
env_types = {'dm', 'atari', 'gym', 'unity'}
env_fn_mappings = {'dm': dm_control2gym.make,
                   'atari': make_atari,
                   'gym': gym.make,
                   'unity': UnityEnvironment,}

# adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
## refactored by nov05
def get_env_fn(game, ## could be called "id", "env_id" in other functions
               env_fn_kwargs = None,
               seed=None, 
               rank=None, 
               episode_life=True):
    random_seed(seed)

    ## get env type
    env_type, kwargs = None, dict()
    if game.startswith("unity"):
        env_type = 'unity'
        kwargs.update(env_fn_kwargs)
    elif game.startswith("dm"):
        env_type = 'dm'
        _, domain, task = game.split('-')
        kwargs.update({'domain_name': domain, 'task_name': task})
    elif hasattr(gym.envs, 'atari') and \
        isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv):
        env_type = 'atari'
        kwargs.update({'env_id':game})
    else:
        env_type = 'gym'
        kwargs.update({'id':game})
    env = env_fn_mappings[env_type](**kwargs)

    if env_type!='unity':
        env.seed(seed + rank)
        env = OriginalReturnWrapper(env)
        if env_type=='atari':
            env = wrap_deepmind(env,
                                episode_life=episode_life,
                                clip_rewards=False,
                                frame_stack=False,
                                scale=False)
            obs_shape = env.observation_space.shape
            if len(obs_shape)==3:
                env = TransposeImage(env)
                env = FrameStack(env, 4)
    return lambda:env, env_type ## return the env_fn

class OriginalReturnWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.total_rewards = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.total_rewards += reward
        if done:
            info['episodic_return'] = self.total_rewards
            self.total_rewards = 0
        else:
            info['episodic_return'] = None
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


# The original LayzeFrames doesn't work well
class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self.__array__())

    def __getitem__(self, i):
        return self.__array__()[i]


class FrameStack(FrameStack_):
    def __init__(self, env, k):
        FrameStack_.__init__(self, env, k)

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


## The original one in baselines is really bad
## baselines\baselines\common\vec_env\dummy_vec_env.py    
class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns] ## create envs
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        data = []
        for i in range(self.num_envs):
            ## info = e.g. {'episodic_return': None}
            obsv, revw, done, info = self.envs[i].step(self.actions[i])
            if done:
                obsv = self.envs[i].reset()
            data.append([obsv, revw, done, info])
        obsvs, revws, dones, infos = zip(*data)
        return obsvs, np.asarray(revws), np.asarray(dones), infos

    def reset(self):
        ## reset all envs, and return next_states
        return [env.reset() for env in self.envs]

    def close(self):
        ## close all envs
        [env.close() for env in self.envs] 


class UnityVecEnv(VecEnv):
    def __init__(self, envs=None, env_fns=None, train_mode=False):
        if envs:
            self.envs = envs ## envs are the same type
        else:
            self.envs = [fn() for fn in env_fns]
        self.train_mode = train_mode
        
        env = self.envs[0]
        self.brain_name = env.brain_names[0]
        brain = env.brains[self.brain_name]
        self.action_size = brain.vector_action_space_size

        ## reset envs
        info = [env.reset(train_mode=train_mode)[self.brain_name] for env in self.envs][0] 
        self.num_agents = len(info.agents)
        self.actions = None

        self.num_envs = len(self.envs)
        ## tranlate Unity ML-Agents spaces to gym spaces for VecEnv compatibility 
        observation_space = Box(float('-inf'), float('inf'), (brain.vector_observation_space_size,), np.float64)
        action_space = Box(-1.0, 1.0, (brain.vector_action_space_size,), np.float32)
        VecEnv.__init__(self, self.num_envs, observation_space, action_space)

    def step_async(self, actions): ## VecEnv downward func
        self.actions = actions

    def step_wait(self): ## VecEnv downward func
        data = []
        for i in range(self.num_envs):
            env_info = self.envs[i].step(self.actions[i])[self.brain_name]
            obsvs, revws, local_dones, env_infos = env_info.vector_observations, env_info.rewards, env_info.local_done, env_info
            ## remove this logic. one unity env can have multiple agents. 
            ## we don't want to reset the env for one agent is done.  
            # if np.any(dones): ## there are multiple agents
            #     obsvs = self.envs[i].reset(train_mode=self.train_mode)
            data.append([obsvs, revws, local_dones, env_infos])
        next_states, rewards, dones, infos = zip(*data)
        return next_states, np.asarray(rewards), np.asarray(dones), infos

    def reset(self, train_mode=False):
        self.train_mode = train_mode
        return [env.reset(train_mode=self.train_mode)[self.brain_name] for env in self.envs]

    def close(self):
        [env.close() for env in self.envs]


class Task:
    def __init__(self,
                 game, ## gym or unity game, called "id" or "env_id" in other funcs
                 num_envs=1,
                 env_fn_kwargs=dict(),
                 envs=None, ## pre-created envs
                 single_process=True,
                 log_dir=None,
                 episode_life=True,
                 seed=None):
        '''
        20240310 added logic for unity
        '''
        self.game = game
        self.num_envs = num_envs
        self.env_fn_kwargs = env_fn_kwargs
        self.envs = envs
        self.single_process = single_process
        self.log_dir = log_dir
        self.episode_life = episode_life
        self.seed = seed
        
        if not seed:
            # seed = np.random.randint(int(1e9)) ## nov05
            seed = np.random.randint(0, 2**31) ## nov05
        if log_dir:
            mkdir(log_dir)

        ## get env_fns
        self.env_type = None
        if envs:
            self.num_envs = len(envs)
            if isinstance(envs[0], UnityEnvironment):
                self.env_type = 'unity'
        else:
            self.num_envs = num_envs
            if game.startswith('unity'):
                self.env_type = 'unity'
            self.env_fns = []
            for i in range(self.num_envs):
                if self.env_type=='unity':
                    self.env_fn_kwargs.update({'worker_id':i})
                env_fn, self.env_type = get_env_fn(game, env_fn_kwargs=self.env_fn_kwargs, 
                    seed=seed, rank=i, episode_life=episode_life)
                self.env_fns.append(env_fn)

        ## create envs
        Wrapper, wrapper_kwargs = None, None
        if self.env_type=='unity':
            if single_process:
                Wrapper = UnityVecEnv
            else:
                Wrapper = SubprocVecEnv
                # raise NotImplementedError("Multiprocessing is not implemented for Unity envs.")
        else:
            if single_process:
                Wrapper = DummyVecEnv
            else:
                Wrapper = SubprocVecEnv
        if self.envs:
            wrapper_kwargs = {'envs': self.envs}
        else:
            wrapper_kwargs = {'env_fns': self.env_fns}
        self.envs_wrapper = Wrapper(**wrapper_kwargs)
            
        self.observation_space = self.envs_wrapper.observation_space
        self.state_dim = int(np.prod(self.observation_space.shape))
        self.action_space = self.envs_wrapper.action_space
        if isinstance(self.action_space, Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            assert 'unknown action space'
        
    def reset(self, train_mode=False):
        if self.env_type=='unity':
            return self.envs_wrapper.reset(train_mode=train_mode)
        else: 
            return self.envs_wrapper.reset()
        
    def step(self, actions):
        if isinstance(self.action_space, Box): 
            actions = [np.clip(a, self.action_space.low, self.action_space.high) for a in actions]
        return self.envs_wrapper.step(actions)
    
    def close(self):
        return self.envs_wrapper.close()