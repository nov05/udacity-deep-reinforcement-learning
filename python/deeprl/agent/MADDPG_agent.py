#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
## 1. Derives from DDPG_agent.py by nov05 in Oct 2024
##    class inheritance: MADDPGAgent -> UnityBaseAgent -> BaseAgent
## 2. Multi-environments are used for parallel training here, which can add diversity to the experiences.
##    Additionally, asynchronous stepping may help speed up the training process.
## 3. I'm borrowing the 'brain' concept from Unity environments, so I refer to the control unit of an agent as a 'brain.' 
##    This includes components like neural networks and replay buffers. For example, if there are 3 environments with 2 agents 
##    per environment, a total of 2 agent brains will be created. Each brain consists of 4 networks 
##    (a local actor, local critic, target actor, and target critic) and 1 replay buffer. It controls 1 agent, 
##    processes the agent's observations, and generates actions in each environment, while independently updating 
##    its 4 neural networks. This means that an agent brain receives observations from all 3 environments and 
##    generates corresponding actions for each. The 'state, reward, action, next state' (x, r, a, x') sequences 
##    from all environments are stored in the agent's own replay buffer. However, during training, an agent brain 
##    can access the observations and actions of other agent brains. (https://arxiv.org/pdf/1706.02275)



import torch.nn.functional as F
from functools import reduce

## local imports
from ..network import *
from ..component import *
from .BaseAgent import *
from unityagents.exception import UnityActionException



class MADDPGAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        ## if it is eval task and config.num_workers==0, then config.task is config.task_eval  
        ## task.env_type=='unity' indicates it is a Unity env.
        self.task = config.task if config.task is not None else config.task_fn()  ## task (with envs)
        self.num_agents = self.task.envs_wrapper.num_agents  ## for convenience
        self.networks = [config.network_fn() for _ in range(self.num_agents)]  ## local neural network (actor and critic)
        self.network = self.networks[0]  ## for logging logic in misc.py
        self.target_networks = [config.network_fn() for _ in range(self.num_agents)] ## target neural network (actor and critic)
        self.replays = [config.replay_fn() for _ in range(self.num_agents)] ## a central replay buffer for all agents
        self.random_process = config.random_process_fn()  ## random states aka. noise
        self.states = None  ## if None, reset task to states
        ## some envs have to be reset after certain steps, e.g. unity tennis 5000 (self.config.task_name=='unity-tennis')
        self.last_reset_at_step = 0  ## use with config.reset_interval
        self.actor_update_counter = [0] * self.num_agents
        
        ## initialize target networks with local networks
        for local_network, target_network in zip(self.networks, self.target_networks):
            target_network.load_state_dict(local_network.state_dict()) 


    def step(self):
        '''
        self.states, actions, rewards, next_states, dones, infos are for task, with shape [num_envs, num_agents, *dim].
        states_, actions_, rewards_, next_states_ dones_ are for replay buffer, with shape [num_envs, num_agents, *dim].
        e.g. Unity Tennis, 3 envs, 2 agents per env:
            shapes of the task (3, 2, 24) (3, 2, 2) (3, 2) (3, 2, 24) (3, 2) (3,)
            shapes for the replay buffer (3, 2, 24) (3, 2, 2) (3, 2) (3, 2, 24) (3, 2)
        '''
        ## reset the task (envs)
        if ((self.config.reset_interval is not None) 
                and (self.total_steps-self.last_reset_at_step>=self.config.reset_interval)
            or self.states is None):
            self.random_process.reset_states() ## denoted as 𝒩 in the paper
            self.states = self._reset_task(self.task) ## [num_envs, num_agents, state dims]
            self.last_reset_at_step = self.total_steps
        states_ = self._reshape_for_network(self.states, keep_dim=3) ## no change in dims in this case
        states_ = self.config.state_normalizer(states_) ## do nothing in this case

        ## step
        if self.total_steps < self.config.warm_up: ## generate random actions
            actions = self._sample_actions()  ## [num_envs, num_agents, action dims]
            actions_ = self._reshape_for_network(actions, keep_dim=3)  ## no change in dims in this case
        else:  ## get actions from the local network
            actions_ = []
            for i in range(self.num_agents):
                self.networks[i].eval()
                with torch.no_grad():
                    ## get action from local actor; action_i shape [num_envs, action dims]
                    action_i = to_np(self.networks[i](tensor(states_).transpose(0, 1)[i])) 
                    # check_tensor('action_i', self.networks[i](tensor(states_).transpose(0, 1)[i]))
                    ## add noise, denoted by 𝒩_t in the paper
                    action_i += (
                        self.random_process.sample()                                      ## add noise
                        # * (1/((self.total_episodes+1)**self.config.noise_decay_factor))   ## with decay
                        * 0.22                                                             ## constant
                       )  ## add noise
                    actions_.append(action_i[:,np.newaxis,:])
                self.networks[i].train()
            actions_ = np.concatenate(actions_, axis=1)  ## [num_envs, num_agents, action dims]
            actions = self._reshape_for_task(self.task, actions_)  ## no dim change in this case
        ## task will clip actions when step
        next_states, rewards, dones, infos = self.task.step(actions)

        ## tidy up trajectory data for the replay buffer
        actions_ = np.clip(actions_, self.task.action_space.low, self.task.action_space.high)
        next_states_ = self._reshape_for_network(next_states, keep_dim=3) ## no change in dims in this case 
        next_states_ = self.config.state_normalizer(next_states_) ## do nothing in this case
        rewards_ = self._reshape_for_network(rewards, keep_dim=2) ## no change in dims in this case
        rewards_ = self.config.reward_normalizer(rewards_) ## do nothing in this case
        dones_ = self._reshape_for_network(dones, keep_dim=2) ## no change in dims in this case
        
        ## update replay buffers with new (s, a, r, s_prim) trajectories
        for i in range(self.num_agents):
            self.replays[i].feed(dict(
                state=states_, 
                action=actions_, 
                reward=rewards_, 
                next_state=next_states_, 
                mask=1-np.asarray(dones_, dtype=np.int32),
            ))
        self.states = next_states

        ## sample config.mini_batch_size (denoted as S) of transition sequences from the replay buffer
        ## to update neural networks for each agent; mini_batch_size is set in config.py
        if (self.replays[0].size() >= self.config.warm_up 
        and self.total_steps%self.config.replay_interval == 0):  ## replay every interval steps

            for agent_index in range(self.num_agents):
                transitions = self.replays[agent_index].sample()
                ## convert to tensor and move to the device, change shape to [num_agents, mini_batch_size, *dims]
                states_ = tensor(transitions.state).transpose(0, 1) 
                actions_ = tensor(transitions.action).transpose(0, 1) 
                rewards_ = tensor(transitions.reward).unsqueeze(-1).transpose(0, 1) 
                next_states_ = tensor(transitions.next_state).transpose(0, 1)
                masks_ = tensor(transitions.mask).unsqueeze(-1).transpose(0, 1) 
                sampling_probs_ = tensor(transitions.sampling_prob).unsqueeze(-1)  ## [mini_batch_size, 1]
                sample_weights_ = 1.0 / (sampling_probs_ * self.replays[agent_index].size())  ## [mini_batch_size, 1]

                ## the networks can process data with dimension [mini_batch_size, *network_input_dims]
                ## no backprobagation for the target network; it will be updated from local later
                with torch.no_grad():
                    ## target actors forward
                    ## input x′n_j, output a′n with shape [mini_batch_size, action_length]; 'j' denotes 'replay sample'
                    ## a_target (a′) is a list of actions (a′n) with shape of [mini_batch_size, num_agents*action_length]
                    ## add noise to smooth the critic fit
                    a_target = (
                        torch.cat([
                            torch.clamp(
                                (self.target_networks[i].actor(next_states_[i])
                                + tensor(self.random_process.sample())                           ## add noise
                                # * (1/((self.total_episodes+1)**self.config.noise_decay_factor))  ## with decay
                                * 0.25                                                            ## constant factor
                                ), self.task.action_space.low[i], self.task.action_space.high[i]
                            ) for i in range(self.num_agents)], dim=1
                        )
                    )
                    ## get target Q-value; target critic forward
                    ## input (x′_j, a′1, ..., a′n), output shape [mini_batch_size, 1]            
                    q_target_i = reduce(lambda x, y: torch.minimum(x, y), [
                        rewards_[i] + masks_[i]*self.config.discount*q.detach() for q in
                        self.target_networks[i].critic(
                            next_states_.transpose(0, 1).reshape(self.config.mini_batch_size, -1), 
                            a_target)
                    ])

                ## get local Q-value; local critic forward
                ## input (x_j, a1_j, ..., an_j), output shape [mini_batch_size, 1]
                q_critic_i = self.networks[i].critic(
                    states_.transpose(0, 1).reshape(self.config.mini_batch_size, -1), 
                    actions_.transpose(0, 1).reshape(self.config.mini_batch_size, -1))
                ## get squared TD-error, for replay priority updating 
                se_loss_i = reduce(lambda x, y: torch.add(x, y), 
                    [F.mse_loss(q_target_i, q, reduction='none') for q in q_critic_i]
                ) 
                ## get local critic loss
                ## MSE, both input shapes [mini_batch_size, 1], output shape (1,)
                critic_loss_i = torch.mean(se_loss_i*sample_weights_, dim=0) 
                # check_tensor('critic_loss_i', critic_loss_i)  ## check NaNs, Infs
                ## local critic backpropagation
                self.networks[agent_index].critic_opt.zero_grad() 
                critic_loss_i.backward()
                # torch.nn.utils.clip_grad_norm_(self.networks[i].critic_body.parameters(), max_norm=10.0)
                self.networks[agent_index].critic_opt.step()  ## optimizer step
                # check_network_params(f'critic[{agent_index}]', self.networks[agent_index].critic_body)

                ## update sampling priorities
                with torch.no_grad():
                    priorities_i = se_loss_i.sqrt().squeeze().cpu().numpy()  ## (mini_batch_size,)
                self.replays[agent_index].update_priorities(
                    list(zip(*[transitions.idx, priorities_i])))
                
                ## update local actor
                self.actor_update_counter[agent_index] += 1
                if self.actor_update_counter[agent_index] >= self.config.actor_network_update_freq:

                    ## local actor forward
                    ## input ok_j, output shape [mini_batch_size, action_length]
                    ## 'k' denotes 'policy emsemble', which is not in use here
                    a_i = self.networks[agent_index].actor(states_[agent_index])  
                    a_i = torch.clamp(a_i, 
                                      self.task.action_space.low[agent_index], 
                                      self.task.action_space.high[agent_index])  ## clip action
                    ## (a1_j, ..., a_i, ..., an_j), shape [mini_batch_size, num_agents*action_length]
                    a = torch.cat([actions_[j] if j!=agent_index else a_i 
                                   for j in range(self.num_agents)], dim=1) 
                    ## local actor loss
                    ## input (x_j, a1_j, ..., a_i, ..., an_j), output shape (1,)
                    actor_loss_i = -self.networks[agent_index].critic(
                        states_.transpose(0,1).reshape(self.config.mini_batch_size, -1), 
                        a)[0].mean(dim=0) 
                    # check_tensor('actor_loss', actor_loss_i)  ## check NaNs and Infs  
                    ## local actor backpropagation
                    self.networks[agent_index].actor_opt.zero_grad()  
                    actor_loss_i.backward()
                    # torch.nn.utils.clip_grad_norm_(self.networks[i].actor_body.parameters(), max_norm=10.0)
                    self.networks[agent_index].actor_opt.step()  ## optimizer step
                    # check_network_params(f'actor[{agent_index}]', self.networks[agent_index].actor_body)

                    self.actor_update_counter[agent_index] = 0  ## reset

                ## update target network from local
                soft_update_network(self.target_networks[agent_index], self.networks[agent_index], 
                                    self.config.target_network_mix)
                # check_network_params(f'target_networks[{agent_index}]', self.target_networks[agent_index])

        ## Some environments have a fixed number of steps per episode, like Unity’s Reacher V2, 
        ## while others don’t, such as Unity Tennis. Still, this setup helps with monitoring the training process.
        ## When latter, this works as getting a total (config.max_episodes * self.task.num_envs) of episodes.
        for done,info in zip(dones,infos): ## check whether an episode is done in each env
            if np.any(done):
                self.episodic_returns_all_envs.append(info['episodic_return'])
        if len(self.episodic_returns_all_envs)>=self.task.num_envs:
            self.episode_done_all_envs = True  ## used in func 'run_episodes()' in misc.py
            self.record_online_return(self.episodic_returns_all_envs, 
                                      by_episode=self.config.by_episode)  ## log train returns
            self.episodic_returns_all_envs = []
            self.total_episodes += 1
        self.total_steps += 1


    def eval_step(self, states):
        if states is None:
            raise Exception("⚠️ \"states\" is None.")
        states_ = self._reshape_for_network(states, keep_dim=3)  ## no change in dims in this case, [num_envs, num_agents, state dims]
        self.config.state_normalizer.set_read_only()
        states_ = self.config.state_normalizer(states_)  ## do nothing in this case
        states_ = tensor(states_).transpose(0,1)  ## [num_agents, num_envs, state dims]
        actions_ = []
        for i in range(self.num_agents):
            ## get actions from the local actors
            self.networks[i].eval()
            with torch.no_grad():
                action_i = to_np(self.networks[i](states_[i]))  
                actions_.append(action_i[:,np.newaxis,:])
            self.networks[i].train()
        self.config.state_normalizer.unset_read_only()
        actions = self._reshape_for_task(self.eval_task, actions_)  ## ## [num_agents, num_envs, action dims]
    
        return actions