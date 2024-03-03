# import os
# os.add_dll_directory(r"C:/Users/guido/.mujoco/mjpro150/bin")
# os.add_dll_directory(r"C:/Users/guido/.mujoco/mujoco-py-1.50.1.68/mujoco_py")
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
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

# try:
#     import roboschool
# except ImportError as e:
#     print(e)
#     print("You are probably using Windows. Roboschool doesn't work on Windows.")
#     pass

import sys
import warnings
if not sys.warnoptions:  # allow overriding with `-W` option
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='runpy')
gym.logger.set_level(40)   



# adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
def make_env(env_id, seed, rank, episode_life=True):
    def _thunk():
        random_seed(seed)
        if env_id.startswith("dm"):
            import dm_control2gym
            _, domain, task = env_id.split('-')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)
        env = OriginalReturnWrapper(env)
        if is_atari:
            env = wrap_deepmind(env,
                                episode_life=episode_life,
                                clip_rewards=False,
                                frame_stack=False,
                                scale=False)
            obs_shape = env.observation_space.shape
            if len(obs_shape)==3:
                env = TransposeImage(env)
            env = FrameStack(env, 4)

        return env

    return _thunk


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
        self.envs = [fn() for fn in env_fns]
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


class MLAgentsVecEnv(VecEnv):
    def __init__(self, env, train_mode=False):
        self.envs = [env] ## one env is imported in this case
        self.train_mode = train_mode

        self.brain_name = env.brain_names[0]
        brain = env.brains[self.brain_name]
        self.action_size = brain.vector_action_space_size
        info = env.reset(train_mode=train_mode)[self.brain_name] ## reset env
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
            ## remove this logic. one unity env has multiple agents. 
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
                 name,
                 num_envs=1,
                 env=None, ## pass a pre-created env
                 is_mlagents=False,
                 single_process=True,
                 log_dir=None,
                 episode_life=True,
                 seed=None):
        self.name = name
        self.num_envs = num_envs
        self.is_mlagents = is_mlagents

        if not seed:
            seed = np.random.randint(int(1e9))
        if log_dir:
            mkdir(log_dir)
        if not env:
            env_fns = [make_env(name, seed, i, episode_life) for i in range(self.num_envs)]

        if is_mlagents: ## Unity ML-Agents
            self.envs_wrapper = MLAgentsVecEnv(env)
        else:
            if single_process:
                Wrapper = DummyVecEnv
            else:
                Wrapper = SubprocVecEnv
            self.envs_wrapper = Wrapper(env_fns)
            
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
        if self.is_mlagents:
            return self.envs_wrapper.reset(train_mode=train_mode)
        else: 
            return self.envs_wrapper.reset()
        
    def step(self, actions):
        if isinstance(self.action_space, Box): 
            actions = [np.clip(a, self.action_space.low, self.action_space.high) for a in actions]
        return self.envs_wrapper.step(actions)
    
    def close(self):
        return self.envs_wrapper.close()


    ## This might be helpful for custom env debugging
    # env_dict = gym.envs.registration.registry.env_specs.copy()
    # for item in env_dict.items():
    #     print(item)


## nov05, in the dir "./python", run "python -m deeprl.component.envs" in terminal
import pandas as pd
if __name__ == '__main__':

    option = '11' ## 0, 10, 11
    if option[0]=='0':
        task = Task('Hopper-v2', num_envs=10, single_process=True) ## multiprocessing doesn't work in Windows
        state = task.reset()
        for _ in range(100):
            actions = [np.random.rand(task.action_space.shape[0])] * task.num_envs
            _, _, dones, _ = task.step(actions)
            if np.any(dones):
                print(dones)
        task.close()

    elif option[0]=='1':
        from unityagents import UnityEnvironment
        # file_name = '..\data\Reacher_Windows_x86_64_1\Reacher.exe'
        file_name = '..\data\Reacher_Windows_x86_64_20\Reacher.exe'
        env = UnityEnvironment(file_name=file_name, no_graphics=False)

        if option[1]=='0':
            brain_name = env.brain_names[0]
            brain = env.brains[brain_name]
            env_info = env.reset(train_mode=False)[brain_name]     # reset the environment 
            num_agents = len(env_info.agents)
            action_size = brain.vector_action_space_size
            states = env_info.vector_observations                  # get the current state (for each agent)
            scores = np.zeros(num_agents)                          # initialize the score (for each agent)
            while True:
                actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
                actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
                env_info = env.step(actions)[brain_name]           # send all actions to tne environment
                rewards = env_info.rewards                         # get reward (for each agent)
                dones = env_info.local_done                        # see if episode finished
                scores += env_info.rewards                         # update the score (for each agent)
                if np.any(dones):                                  # exit loop if episode finished
                    print("An agent finished an episode!")
                    break
            print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
            env.close()

        elif option[1]=='1':
            num_envs, env_id = 1, 0
            task = Task('Reacher-v2', num_envs=num_envs, 
                        env=env, is_mlagents=True, single_process=True)
            scores = np.zeros(task.envs_wrapper.num_agents) 
            for i in range(10000):
                actions = [np.random.randn(task.envs_wrapper.num_agents, task.action_space.shape[0])] * task.num_envs
                _, rewards, dones, infos = task.step(actions)
                scores += rewards[env_id]
                if np.any(rewards[env_id]):
                    print(pd.DataFrame([rewards[env_id], scores], index=['rewards','scores']))
                    # print("max reached:", infos[env_id].max_reached)
                if np.any(dones[env_id]): ## if any agent finishes an episode
                    print("An agent finished an episode!")
                    break
            print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
            task.close()