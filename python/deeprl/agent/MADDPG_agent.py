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
## local imports
from ..network import *
from ..component import *
from .BaseAgent import *



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
        self.replays = [config.replay_fn() for _ in range(self.num_agents)]  ## replay buffers 
        self.random_process = config.random_process_fn()  ## random states aka. noise
        self.states = None
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
        if self.states is None:
            self.random_process.reset_states() ## denoted as 𝒩 in the paper
            self.states = self._reset_task(self.task) ## [num_envs, num_agents, state dims]
        states_ = self._reshape_for_network(self.states, keep_dim=3) ## do nothing in this case
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
                    # if np.isnan(action_i).any(): raise RuntimeError(f"⚠️ Actor[{i}] created NaN action.")
                    ## add noise with decay, denoted by 𝒩_t in the paper
                    action_i += self.random_process.sample() * (1/(self.total_episodes+1)**0.3)
                    actions_.append(action_i[:,np.newaxis,:])
                self.networks[i].train()
            actions_ = np.concatenate(actions_, axis=1)  ## [num_envs, num_agents, action dims]
            actions = self._reshape_for_task(self.task, actions_)  ## no dim change in this case
        ## task will clip actions when step
        next_states, rewards, dones, infos = self.task.step(actions)

        ## tidy up trajectory data for the replay buffer
        actions_ = np.clip(actions_, self.task.action_space.low, self.task.action_space.high)
        next_states_ = self._reshape_for_network(next_states, keep_dim=3) ## no change in dims in this case 
        next_states_ = self.config.state_normalizer(next_states_) ## no change in dims in this case
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
        ## to update neural networks for each agent
        if self.replays[0].size() >= self.config.warm_up \
        and self.total_steps%self.config.replay_interval == 0:  ## replay every interval steps
            
            for i in range(self.num_agents):

                ## mini_batch_size is set in config.py
                transitions = self.replays[i].sample()  
                ## convert to tensor and move to the device, change shape to [num_agents, mini_batch_size, *dims]
                states_ = tensor(transitions.state).transpose(0, 1) 
                actions_ = tensor(transitions.action).transpose(0, 1)
                rewards_ = tensor(transitions.reward).unsqueeze(-1).transpose(0, 1)
                next_states_ = tensor(transitions.next_state).transpose(0, 1)
                masks_ = tensor(transitions.mask).unsqueeze(-1).transpose(0, 1)
                sampling_probs_ = tensor(transitions.sampling_prob).unsqueeze(-1).transpose(0, 1)
                sample_weights_ = 1.0 / (sampling_probs_ * self.replays[i].size())  ## make sure it is not Inf

                ## the networks can process data with dimension [mini_batch_size, *network_input_dim]
                ## no backprobagation for the target network; it will be updated from local later
                with torch.no_grad():
                    ## target actor forward
                    ## input (ok_j), output a′k with shape [mini_batch_size, *action_dims]
                    ## a_target is a list of actions; change shape to [mini_batch_size, num_agents*action_vector_length]
                    a_target = tensor([])
                    for k in range(self.num_agents):
                        a_target_k = self.target_networks[k].actor(next_states_[k])
                        # a_target_k += tensor(self.random_process.sample() * (1/(self.total_episodes+1)**0.5))  ## add noise
                        a_target_k = torch.clamp(a_target_k,  
                            self.task.action_space.low[k], self.task.action_space.high[k])
                        a_target = torch.cat([a_target, a_target_k], dim=1)
                    
                    ## target critic forward
                    ## input (x′_j, a′1, ..., a′n), output shape [mini_batch_size, 1]
                    q_target = self.target_networks[i].critic(
                        next_states_.transpose(0, 1).reshape(self.config.mini_batch_size, -1), 
                        a_target)
                    ## multiply λ-discount rate, add reward
                    y = rewards_[i] + masks_[i]*self.config.discount*q_target  
                    y = y.detach() 
                
                ## local critic forward
                ## input (x_j, a1_j, ..., an_j), output shape [mini_batch_size, 1]
                q_critic = self.networks[i].critic(
                    states_.transpose(0, 1).reshape(self.config.mini_batch_size, -1), 
                    actions_.transpose(0, 1).reshape(self.config.mini_batch_size, -1))
                ## local critic loss
                se_loss = F.mse_loss(y, q_critic, reduction='none')  
                critic_loss = torch.mean(torch.sum(se_loss*sample_weights_))  ## MSE
                # if torch.isnan(critic_loss): raise RuntimeError("⚠️ Critic loss is NaN.")
                # if torch.isinf(critic_loss): raise RuntimeError("⚠️ Critic loss is Inf.")
                ## local critic backpropagation
                self.networks[i].critic_opt.zero_grad() 
                critic_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.networks[i].critic_body.parameters(), max_norm=1.0)
                self.networks[i].critic_opt.step()  ## optimizer step
                # check_network_params(f'critic[{i}]', self.networks[i].critic_body)

                ## update sampling priorities
                priorities = se_loss.detach().sqrt().squeeze().cpu().numpy()
                self.replays[i].update_priorities(list(zip(*[transitions.idx, priorities])))
            
                ## update local actor
                self.actor_update_counter[i] += 1
                if self.actor_update_counter[i] >= self.config.actor_update_freq:

                    ## local actor forward
                    ## input ok_j, output shape [mini_batch_size, action_vector_length]
                    a_i = self.networks[i].actor(states_[i])  
                    a_i = torch.clamp(a_i, self.task.action_space.low[i], self.task.action_space.high[i])  ## clip action
                    ## (a1_j, ..., a_i, ..., an_j), shape [mini_batch_size, num_agents*action_vector_length]
                    a = torch.cat([actions_[j] if j!=i else a_i for j in range(self.num_agents)], dim=1) 
                    ## local actor loss
                    ## input (x_j, a1_j, ..., a_i, ..., an_j)
                    actor_loss = -self.networks[i].critic(
                        states_.transpose(0,1).reshape(self.config.mini_batch_size, -1), 
                        a).mean(dim=0) 
                    # if torch.isnan(actor_loss): raise RuntimeError("⚠️ Actor loss is NaN.")
                    # if torch.isinf(actor_loss): raise RuntimeError("⚠️ Actor loss is Inf.")
                    ## local actor backpropagation
                    self.networks[i].actor_opt.zero_grad()  
                    actor_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.networks[i].actor_body.parameters(), max_norm=1.0)
                    self.networks[i].actor_opt.step()  ## optimizer step
                    # check_network_params(f'actor[{i}]', self.networks[i].actor_body)

                    self.actor_update_counter[i] = 0  ## reset

                ## update target network from local
                soft_update_network(self.target_networks[i], self.networks[i], 
                                    self.config.target_network_mix)
                # check_network_params(f'target_networks[{i}]', self.target_networks[i])

        ## Some environments have a fixed number of steps per episode, like Unity’s Reacher V2, 
        ## while others don’t, such as Unity Tennis. Still, this setup helps with monitoring the training process.
        for done,info in zip(dones,infos): ## check whether an episode is done in each env
            if np.any(done):
                self.episodic_returns_all_envs.append(info['episodic_return'])
        if len(self.total_episodic_returns)==self.task.num_envs:
            self.episode_done = True
            self.record_online_return(self.total_episodic_returns, 
                                      by_episode=self.config.by_episode)  ## log train returns
            self.total_episodic_returns = []
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
        self.config.state_normalizer.unset_read_only()
        actions = self._reshape_for_task(self.eval_task, actions_)  ## ## [num_agents, num_envs, action dims]
        return actions