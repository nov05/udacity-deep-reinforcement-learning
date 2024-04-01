#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *



class DDPGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task if config.task is not None else config.task_fn()  ## task (with envs)
        self.network = config.network_fn()  ## local neural network (actor and critic)
        self.target_network = config.network_fn()  ## target neural network (actor and critic)
        self.target_network.load_state_dict(self.network.state_dict()) ## initialize target with local
        self.replay = config.replay_fn()  ## replay buffer 
        self.random_process = config.random_process_fn()  ## random states or noise
        self.states = None
        self.total_episodic_returns = []


    def soft_update(self, target, source):
        ## trg = trg*(1-τ) + src*τ
        ## τ is stored in "self.config.target_network_mix"
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               source_param * self.config.target_network_mix)


    def eval_step(self, states):
        if states is None:
            raise Exception("⚠️ \"states\" is None")
        self.config.state_normalizer.set_read_only()
        states = self.config.state_normalizer(states)
        actions = self.network(states)  ## get actions from the local network
        self.config.state_normalizer.unset_read_only()
        return to_np(actions)


    def step(self):
        ## reset the task
        if self.states is None:
            self.random_process.reset_states()
            if self.env_type in ['unity']:
                self.states, _, _, _ = self.task.reset()
            else:
                self.states = self.task.reset()
            self.states = self.config.state_normalizer(self.states)

        ## step
        if self.total_steps < self.config.warm_up: ## get random actions
            if self.env_type in ['unity']: ## one env has multiple agents
                actions = []
                for _ in range(self.task.num_envs):
                    actions.append([self.task.action_space.sample()
                                    for _ in range(self.task.envs_wrapper.num_agents)])
            else: ## one env has one agent
                actions = [self.task.action_space.sample()
                           for _ in range(self.task.num_envs)]
        else: ## get actions from the local network
            actions = to_np(self.network(self.states)) \
                    + self.random_process.sample() ## add noise
        ## task will clip when step. however, actions have to be clipped here for replay buffer
        actions = np.clip(actions, self.task.action_space.low, self.task.action_space.high)
        next_states, rewards, dones, infos = self.task.step(actions)
        next_states = self.config.state_normalizer(next_states)
        rewards = self.config.reward_normalizer(rewards)
        for done,info in zip(dones,infos):
            if np.any(done):
                self.total_episodic_returns.append(info['episodic_return'])

        # update replay buffer
        states_ = self._reshape_for_replay(self.states, keep_dim=2)
        actions_ = self._reshape_for_replay(actions, keep_dim=2)
        rewards_ = self._reshape_for_replay(rewards, keep_dim=1)
        next_states_ = self._reshape_for_replay(next_states, keep_dim=2)
        dones_ = self._reshape_for_replay(dones, keep_dim=1)
        self.replay.feed(dict(
            state=states_, 
            action=actions_, 
            reward=rewards_, 
            next_state=next_states_, 
            mask=1-np.asarray(dones_, dtype=np.int32),
        ))
    
        ## check whether the episode is done
        if len(self.total_episodic_returns)==self.task.num_envs:
            self.episode_done = True
            self.record_online_return(self.total_episodic_returns, 
                                      by_episode=self.config.by_episode)  ## log train returns
            self.total_episodic_returns = []
            self.states = None
            self.total_episodes += 1
        else:
            self.states = next_states
        self.total_steps += 1

        ## sample config.mini_batch_size of transition sequences from the replay buffer
        ## update neural networks
        if self.replay.size() >= self.config.warm_up:
            transitions = self.replay.sample()  ## batch_size is set in config.replay_fn
            states = tensor(transitions.state)
            actions = tensor(transitions.action)
            rewards = tensor(transitions.reward).unsqueeze(-1)
            next_states = tensor(transitions.next_state)
            mask = tensor(transitions.mask).unsqueeze(-1)

            ## the networks can process data with dimension of (bath_size, observation_size)
            ## target actor (policy network) and critic (value network) forward
            phi_next = self.target_network.feature(next_states)
            a_next = self.target_network.actor(phi_next)  ## get action
            q_next = self.target_network.critic(phi_next, a_next)  ## get Q-value
            q_next = self.config.discount * mask * q_next ## multiply λ-discount rate
            q_next.add_(rewards)
            q_next = q_next.detach()

            ## local critic forward
            phi = self.network.feature(states)
            q = self.network.critic(phi, actions)
            ## local critic loss and backpropagate
            value_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()
            self.network.zero_grad()
            value_loss.backward()
            self.network.critic_opt.step()  ## optimizer step

            ## local actor forward
            phi = self.network.feature(states)
            a = self.network.actor(phi)
            ## local actor loss and backpropagation
            policy_loss = -self.network.critic(phi.detach(), a).mean()
            self.network.zero_grad()
            policy_loss.backward()
            self.network.actor_opt.step()  ## optimizer step

            ## update target network
            self.soft_update(self.target_network, self.network)


    ## added by nov05
    def _reshape_for_replay(self, data, keep_dim=2):
        data = np.array(data)
        if len(data.shape)>keep_dim:
            if keep_dim>1:
                data = data.reshape(-1, *data.shape[-keep_dim+1:]).tolist()
            else:
                data = data.reshape(-1).tolist()
        elif len(data.shape)<=keep_dim:
            data = data.tolist()
        return data