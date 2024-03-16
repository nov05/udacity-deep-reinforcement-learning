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
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, \
    CloudpickleWrapper, clear_mpi_env_vars, _flatten_obs
from unityagents.brain import BrainParameters, BrainInfo

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
import multiprocessing as mp
import dm_control2gym
from unityagents import UnityEnvironment, UnityMultiEnvironment

## wrap the class with a func, or Multiprocessing will throw
## "TypeError: cannot pickle '_thread.lock' object"
def make_unity(**kwargs):
    return lambda:UnityEnvironment(**kwargs)

env_types = {'dm', 'atari', 'gym', 'unity'}
env_fn_mappings = {'dm': dm_control2gym.make,
                   'atari': make_atari,
                   'gym': gym.make,
                   'unity': make_unity}
    
# adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
## refactored, func for unity added, by nov05
def get_env_fn(game, ## could be called "id", "env_id" in other functions
               env_fn_kwargs=None,
               seed=None, 
               rank=None, 
               episode_life=True):
    
    ## get env type
    env_type, kwargs = None, dict()
    if game.startswith("unity"):
        env_type = 'unity'
        kwargs.update(env_fn_kwargs)
        if seed: kwargs.update({'seed':seed})
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

    ## create env    
    env = env_fn_mappings[env_type](**kwargs)

    if env_type!='unity':
        random_seed(seed)
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

    return lambda:env, env_type ## return the env as a func

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


def get_unity_spaces(brain_params): 
    """
    tranlate Unity ML-Agents spaces to gym spaces for compatibility with Baselines 
    """
    if brain_params.vector_observation_space_type=='continuous':
        observation_space = Box(float('-inf'), float('inf'), (brain_params.vector_observation_space_size,), np.float64)
    else:
        raise NotImplementedError
    if brain_params.vector_action_space_type=='continuous':
        action_space = Box(-1.0, 1.0, (brain_params.vector_action_space_size,), np.float32)
    else:
        raise NotImplementedError
    return observation_space, action_space


class UnityVecEnv(VecEnv):
    def __init__(self, env_fns=None, train_mode=False):
        self.envs = [fn()() for fn in env_fns]
        self.train_mode = train_mode
        
        env = self.envs[0]
        self.brain_name = env.brain_names[0]
        brain_params = env.brains[self.brain_name]
        self.action_size = brain_params.vector_action_space_size

        ## reset envs
        info = self.reset(train_mode=train_mode)[0]
        self.num_agents = len(info.agents)
        self.actions = None

        self.num_envs = len(self.envs)
        observation_space, action_space = get_unity_spaces(brain_params)
        VecEnv.__init__(self, self.num_envs, observation_space, action_space)

    def step_async(self, actions): ## VecEnv downward func
        self.actions = actions

    def step_wait(self): ## VecEnv downward func
        data = []
        for i in range(self.num_envs):
            env_info = self.envs[i].step(self.actions[i])[self.brain_name]
            obsv, revw, done, info = env_info.vector_observations, env_info.rewards, env_info.local_done, env_info  
            # if np.all(done): ## there are multiple agents in an env
            #     obsv = self.envs[i].reset(train_mode=self.train_mode)
            data.append([obsv, revw, done, info])
        next_states, rewards, dones, infos = zip(*data)
        return next_states, np.asarray(rewards), np.asarray(dones), infos

    def reset(self, train_mode=None):
        ## reset an env, returning AllBrainInfo
        if train_mode: self.train_mode = train_mode
        if not self.train_mode: self.train_mode = False
        return [env.reset(train_mode=self.train_mode)[self.brain_name] for env in self.envs]

    def close(self):
        [env.close() for env in self.envs]


def unity_worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()()
    brain_name = env.brain_names[0]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd=='step':
                ## all_brain_info: type AllBrainInfo, a dict
                ## e.g. {'ReacherBrain': <unityagents.brain.BrainInfo object at 0x0000022605F2D8A0>}
                all_brain_info = env.step(data) 
                info = all_brain_info[brain_name]
                if brain_name in ['ReacherBrain']:
                    ob = info.vector_observations
                else:
                    raise NotImplementedError
                rew, done = info.rewards, info.local_done
                # if done: ## there are multiple agents in one unity env
                #     ob = env.reset()
                remote.send((ob, rew, done, info))
            elif cmd=='reset':
                all_brain_info = env.reset(data)
                info = all_brain_info[brain_name]
                remote.send(info)
                print('🟢 Unity environment has been resetted.')
            elif cmd=='close':
                remote.close()
                break
            elif cmd=='get_brain_params':
                brain_params = env.brains[brain_name] ## brain_params: type class BrainParameters
                ## the original value seems to be <class 'google.protobuf.pyext._message.RepeatedScalarContainer'>
                ## and not serializable
                brain_params.vector_action_descriptions = ['','','','']
                remote.send(brain_params)
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('UnitySubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class UnitySubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, train_mode=False, context='spawn'):
        """
        Arguments:
        env_fns: iterable of callables - functions that create environments to run in subprocesses. 
                 Need to be cloud-pickleable
        """
        self.env_fns = env_fns
        self.train_mode = train_mode
        self.context = context

        self.waiting = False
        self.closed = False
        self.viewer = None
        self.num_envs = len(env_fns)
        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.ps = [ctx.Process(target=unity_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn))) 
                for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
                print(f"🟢 {p.name} has started.")
        for remote in self.work_remotes:
            remote.close()

        ## get brain info
        self.remotes[0].send(('get_brain_params', None))
        brain_params = self.remotes[0].recv()
        self.brain_name = brain_params.brain_name
        observation_space, action_space = get_unity_spaces(brain_params)
        
        ## reset the envs
        infos = self.reset(train_mode=self.train_mode) 
        self.num_agents = len(infos[0].agents)
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    def reset(self, train_mode=None):
        """
        Reset all Unity environments serially
        """
        self._assert_not_closed()
        infos = []
        for remote in self.remotes:
            remote.send(('reset', train_mode))
            infos.append(remote.recv())
        return infos

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def _assert_not_closed(self):
        assert not self.closed, "⚠️ Trying to operate on a UnitySubprocVecEnv after calling close()"

    def __del__(self):
        if not self.closed:
            self.close()


class Task:
    def __init__(self,
                 game, ## gym or unity game, called "id" or "env_id" in other funcs
                 num_envs=1,
                 env_fn_kwargs=dict(), ## for unity envs
                 train_mode=False, ## for unity envs only
                 seeds=None,
                 single_process=True,
                 log_dir=None,
                 episode_life=True):
        '''
        20240310 added logic for unity
        '''
        self.game = game
        self.num_envs = num_envs
        self.env_fn_kwargs = env_fn_kwargs
        self.train_mode = train_mode 
        self.seeds = seeds
        self.single_process = single_process

        self.envs = []
        self.log_dir = log_dir
        self.episode_life = episode_life
        if log_dir:
            mkdir(log_dir)

        ## get env type
        self.env_type = None
        if game.startswith('unity'):
            self.env_type = 'unity'

        ## get seeds
        if not self.seeds:
            if self.env_type=='unity':
                self.seeds = [np.random.RandomState().randint(-2147483648, 2147483647) for _ in range(self.num_envs)]
            else:
                self.seeds = [np.random.RandomState().randint(0, 2**31) for _ in range(self.num_envs)] 

        ## get env_fns
        self.env_fns = []
        for i in range(self.num_envs):
            if self.env_type=='unity':
                self.env_fn_kwargs.update({'worker_id':i})
            env_fn, self.env_type = get_env_fn(self.game, 
                                               env_fn_kwargs=self.env_fn_kwargs, 
                                               seed=self.seeds[i], 
                                               rank=i, 
                                               episode_life=self.episode_life)
            self.env_fns.append(env_fn)

        ## create envs
        Wrapper, wrapper_kwargs = None, None
        if self.env_type=='unity':
            if single_process:
                Wrapper = UnityVecEnv
            else:
                Wrapper = UnitySubprocVecEnv
            wrapper_kwargs = {'env_fns':self.env_fns, 
                              'train_mode':self.train_mode}
        else:
            if single_process:
                Wrapper = DummyVecEnv 
            else:
                Wrapper = SubprocVecEnv
            wrapper_kwargs = {'env_fns':self.env_fns}
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
        
    def reset(self, train_mode=None):
        ## train_mode is for unity envs only
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