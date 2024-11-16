#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import torch
import numpy as np
import torch.multiprocessing as mp
from collections import deque
from skimage.io import imsave
import pickle
import wandb  ## training tracking

## local imports
from ..utils import *
from deeprl.component.envs import get_env_type



class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(tag=config.tag, log_level=config.log_level)
        self.task_ind = 0

        ## the following attrs are added by nov05 in April 2024
        self.env_type = get_env_type(game=self.config.game)   ## for unity envs
        self.states = None
        self.total_steps = 0
        ## some envs have to be reset after certain steps, 
        ## e.g. unity tennis 5000 (self.config.task_name=='unity-tennis')
        self.last_reset_at_step = 0  ## use with config.reset_interval
        self.total_episodes = 0  ## how many episodes have been executed
        ## a task could have multiple environments
        self.episode_done_all_envs = False   ## all envs have done an episode
        self.episodic_returns_all_envs = []

        ## eval control
        self.eval_states = None
        self.eval_total_steps = 0
        self.eval_last_reset_at_step = 0  ## for reset control
        self.eval_total_episodes = 0  ## how many eval episodes have been executed
        # self.eval_episode_done_all_envs = False   ## all eval envs have done an episode
        self.eval_episodic_returns_all_envs = []  ## one episodic returns of all envs


    def close(self):
        try:
            close_obj(self.eval_task)
            print(f"🟢 Eval task {self.eval_task} has been closed.")
        except:
            pass
        try:
            close_obj(self.task)
            print(f"🟢 Task {self.task} has been closed.")
        except:
            pass


    def save(self, filename):
        torch.save(self.network.state_dict(), '%s.model' % (filename))
        with open('%s.stats' % (filename), 'wb') as f:
            pickle.dump(self.config.state_normalizer.state_dict(), f)


    def load(self, filename):
        # state_dict = torch.load('%s.model' % filename, map_location=lambda storage, loc:storage) ## cpu
        state_dict = torch.load('%s.model' % filename)
        self.network.load_state_dict(state_dict)
        with open('%s.stats' % (filename), 'rb') as f:
            self.config.state_normalizer.load_state_dict(pickle.load(f))


    def eval_step(self, state):
        '''Get actions from agents. Do not step the task (the envs) here.'''
        raise NotImplementedError


    def eval_step_task(self):
        '''step the eval task (the envs)'''
        ## reset the eval envs if necessary
        if (((self.config.reset_interval is not None) 
                and (self.eval_total_steps-self.eval_last_reset_at_step>=self.config.reset_interval))
            or self.eval_states is None
            ):
            self.eval_states = self._reset_task(self.eval_task, train_mode=False)  ## reset
            self.eval_last_reset_at_step = self.eval_total_steps

        ## get actions from agents
        actions = self.eval_step(self.eval_states)  
        ## step the eval task (the envs), get observations, rewards, dones, infos
        self.eval_states, _, dones, infos = self.eval_task.step(actions) 
        for done,info in zip(dones,infos):
            if np.any(done):  ## there are multiple agents in one Unity env, hence multiple values in "done"
                self.eval_episodic_returns_all_envs.append(info['episodic_return'])
        self.eval_total_steps += 1


    def eval_episodes(self, by_episode=False):
        '''
        Evaluate {config.eval_episodes} episodes.
        Return the mean as the 'episodic_return_test' score.
        '''
        ## self.config.eval_env is a Task instance with a wrapper instance of a list of env instances
        self.eval_task = self.config.eval_env 
        log_info = f"Step {self.total_steps}, evaluating {self.config.eval_episodes} episodes " + \
                   f"in {self.config.num_workers_eval} environments"
        if by_episode:
            log_info = f"Episode {self.total_episodes}, " + log_info
        self.logger.info(log_info)

        self.eval_states = None
        self.eval_episodic_returns_all_envs = []  ## one episode returns of all envs
        rets = []  ## a list of scores

        ## start evaluating
        for i in range(self.config.eval_episodes):
            print(f"Evaluating episode {i}...")
            while True:
                self.eval_step_task() 
                ## If the task has only one environment, or multiple identical environments with a fixed number of episodes,
                ## then this means all environments are done an episode. Otherwise, it works like the eval task will run a total
                ## of (config.eval_episodes * self.eval_task.num_envs) episodes approximately. 
                if len(self.eval_episodic_returns_all_envs)>=self.eval_task.num_envs:  ## all envs are done
                    rets += self.eval_episodic_returns_all_envs
                    self.eval_episodic_returns_all_envs = []
                    break 
            self.eval_total_episodes += 1  ## simply a counter, not in use so far

        ret = np.mean(rets)
        log_info = f"Step {self.total_steps}, " + \
                   f"episodic_return_test {ret:.2f}" + \
                   f"({np.std(rets)/np.sqrt(len(rets)):.2f})"
        if by_episode:
            log_info = f"Episode {self.total_episodes}, " + log_info
        self.logger.info(log_info)  ## readable logs
        self.logger.add_scalar('episodic_return_test', ret, self.total_steps)  ## tf logs
        if self.config.wandb: wandb.log({'episodic_return_test': ret}, step=self.total_steps)  ## w&b logs

        return {'episodic_return_test': ret,}


    def record_online_return(self, info, offset=0, by_episode=False):
        ## for non-Unity, info is a tuple that includes a dict with key 'episodic_return'
        if isinstance(info, dict):  ## final call 
            ret = info['episodic_return']
            if ret is not None:
                self.logger.add_scalar('episodic_return_train', ret, self.total_steps+offset)
                self.logger.info(f"Step {self.total_steps+offset}, episodic_return_train {ret}")
        elif isinstance(info, tuple):
            for i, info_ in enumerate(info):
                self.record_online_return(info_, i)   ## recursive calls

        ## for Unity, the input is a list of numerics, added by nov05
        ## don't use type() which doesn't recognize numpy.float64 as float.
        elif isinstance(info, list) and (isinstance(info[0], float) or isinstance(info[0], int)): 
            ret = np.mean(info)
            self.logger.add_scalar('episodic_return_train', ret, self.total_steps+offset)  ## tf logs
            log_info = f"Step {self.total_steps+offset}, episodic_return_train {ret}"
            if by_episode: log_info = f"Episode {self.total_episodes}, " + log_info
            self.logger.info(log_info)
            if self.config.wandb: wandb.log({'episodic_return_train': ret}, 
                                            step=self.total_steps+offset)  ## w&b logs
        else:
            raise NotImplementedError
        
        return {'episodic_return_train': ret}


    def switch_task(self):
        config = self.config
        if not config.tasks:
            return
        segs = np.linspace(0, config.max_steps, len(config.tasks) + 1)
        if self.total_steps > segs[self.task_ind + 1]:
            self.task_ind += 1
            self.task = config.tasks[self.task_ind]
            self.states = self.task.reset()
            self.states = config.state_normalizer(self.states)


    def record_episode(self, dir, env):
        mkdir(dir)
        steps = 0
        state = env.reset()
        while True:
            self.record_obs(env, dir, steps)
            action = self.record_step(state)
            state, _, _, info = env.step(action) ## state, reward, done, info
            ret = info[0]['episodic_return']
            steps += 1
            if ret:
                break


    def record_step(self, state):
        raise NotImplementedError


    # For DMControl
    def record_obs(self, env, dir, steps):
        env = env.env.envs[0]
        obs = env.render(mode='rgb_array')
        imsave('%s/%04d.png' % (dir, steps), obs)


    ## added by nov05
    def _reshape_for_network(self, data, keep_dim=2):
        data = np.array(data)
        num_dims = len(data.shape)
        if num_dims>keep_dim:
            if keep_dim>1:
                data = data.reshape(-1, *data.shape[-keep_dim+1:])
            else:
                data = data.reshape(-1)
        # data = data.tolist()
        return data
    

    ## added by nov05
    def _reshape_for_task(self, task, data):
        if self.env_type in ['unity']: ## one env has multiple agents
            data = np.array(data)
            data = data.reshape(task.num_envs, task.envs_wrapper.num_agents, -1)
        return data
    

    ## added by nov05
    def _reset_task(self, task, train_mode=True):
        if self.env_type in ['unity']: ## one env has multiple agents
            states, _, _, _ = task.reset(train_mode=train_mode)  ## observations, rewards, dones, infos
        else:
            states = task.reset()
        return states


    def _sample_actions(self):
        if self.env_type in ['unity']: ## one env has multiple agents
            actions = []
            for _ in range(self.task.num_envs):
                actions.append([self.task.action_space.sample() 
                                for _ in range(self.task.envs_wrapper.num_agents)])
        else: ## one env has one agent
            actions = [self.task.action_space.sample()
                       for _ in range(self.task.num_envs)]
        return actions



class BaseActor(mp.Process):
    STEP = 0
    RESET = 1
    EXIT = 2
    SPECS = 3
    NETWORK = 4
    CACHE = 5

    def __init__(self, config):
        mp.Process.__init__(self)
        self.config = config
        self.__pipe, self.__worker_pipe = mp.Pipe()

        self._state = None
        self._task = None
        self._network = None
        self._total_steps = 0
        self.__cache_len = 2

        if not config.async_actor:
            self.start = lambda: None
            self.step = self._sample
            self.close = lambda: None
            self._set_up()
            self._task = config.task_fn()

    def _sample(self):
        transitions = []
        for _ in range(self.config.sgd_update_frequency):
            transition = self._transition()
            if transition is not None:
                transitions.append(transition)
        return transitions

    def run(self):
        self._set_up()
        config = self.config
        self._task = config.task_fn()

        cache = deque([], maxlen=2)
        while True:
            op, data = self.__worker_pipe.recv()
            if op == self.STEP:
                if not len(cache):
                    cache.append(self._sample())
                    cache.append(self._sample())
                self.__worker_pipe.send(cache.popleft())
                cache.append(self._sample())
            elif op == self.EXIT:
                self.__worker_pipe.close()
                return
            elif op == self.NETWORK:
                self._network = data
            else:
                raise NotImplementedError

    def _transition(self):
        raise NotImplementedError

    def _set_up(self):
        pass

    def step(self):
        self.__pipe.send([self.STEP, None])
        return self.__pipe.recv()

    def close(self):
        self.__pipe.send([self.EXIT, None])
        self.__pipe.close()

    def set_network(self, net):
        if not self.config.async_actor:
            self._network = net
        else:
            self.__pipe.send([self.NETWORK, net])
