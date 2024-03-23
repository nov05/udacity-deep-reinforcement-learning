#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
# import torchvision



class DDPGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()  ## task (with envs) function
        self.network = config.network_fn()  ## neural network function
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()  ## replay butter function
        self.random_process = config.random_process_fn()  ## noise function
        self.total_steps = 0
        self.state = None


    def soft_update(self, target, source):
        ## trg = trg*(1-τ) + src*τ
        ## τ is stored in self.config.target_network_mix
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               source_param * self.config.target_network_mix)


    def eval_step(self, state):
        if not state:
            raise Exception("\"state\" is None")
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        action = self.network(state)
        self.config.state_normalizer.unset_read_only()
        return to_np(action)


    def step(self):
        ## get next_state from action
        if self.state is None:
            self.random_process.reset_states()
            if self.env_type in ['unity']:
                self.state, _, _, _ = self.task.reset()
            else:
                self.state = self.task.reset()
            self.state = self.config.state_normalizer(self.state)

        if self.total_steps < self.config.warm_up:
            if self.env_type in ['unity']: ## one env has multiple agents
                action = []
                for _ in range(self.task.num_envs):
                    action.append([self.task.action_space.sample()
                                   for _ in range(self.task.envs_wrapper.num_agents)])
            else:
                action = [self.task.action_space.sample()
                          for _ in range(self.task.num_envs)]
        else:
            action = self.network(self.state)
            action = to_np(action)
            action += self.random_process.sample() ## add noise
            action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        next_state, reward, done, info = self.task.step(action)
        next_state = self.config.state_normalizer(next_state)
        reward = self.config.reward_normalizer(reward)
        self.record_online_return(info)  ## log train returns

        # update replay buffer
        state_ = self._reshape_for_replay(self.state, keep_dim=2)
        action_ = self._reshape_for_replay(action, keep_dim=2)
        reward_ = self._reshape_for_replay(reward, keep_dim=1)
        next_state_ = self._reshape_for_replay(next_state, keep_dim=2)
        done_ = self._reshape_for_replay(done, keep_dim=1)
        self.replay.feed(dict(
            state=state_, #self.state,
            action=action_, #action,
            reward=reward_, #reward,
            next_state=next_state_, #next_state,
            mask=1-np.asarray(done_, dtype=np.int32),
        ))
    
        if np.any(done_):
            self.random_process.reset_states()
        self.state = next_state
        self.total_steps += 1

        ## sample batch_size of transition sequences from the replay buffer
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


    def _reshape_for_replay(self, data, keep_dim=2):
        data = np.array(data)
        if len(data.shape)>keep_dim:
            if keep_dim>1:
                data = data.reshape(-1, data.shape[-1]).tolist()
            else:
                data = data.reshape(-1).tolist()
        elif len(data.shape)<=keep_dim:
            data = data.tolist()
        return data
