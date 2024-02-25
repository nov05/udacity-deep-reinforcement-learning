import numpy as np

## local imports 
from deeprl import *
from unityagents import UnityEnvironment


def ddpg_continuous(env=None, is_mlagents=True, **kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game, env=env, is_mlagents=is_mlagents)
    config.eval_env = config.task_fn()
    config.max_steps = 1000 #int(1e6)
    config.eval_interval = int(1e4)
    config.eval_episodes = 20

    config.network_fn = lambda: DeterministicActorCriticNet(
        config.state_dim, config.action_dim,
        actor_body=FCBody(config.state_dim, (400, 300), gate=F.relu),
        critic_body=FCBody(config.state_dim + config.action_dim, (400, 300), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    config.replay_fn = lambda: UniformReplay(memory_size=int(1e6), batch_size=100)
    config.discount = 0.99
    config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
        size=(config.action_dim,), std=LinearSchedule(0.2))
    config.warm_up = int(1e4)
    config.target_network_mix = 5e-3
    run_steps(DDPGAgent(config))


env = UnityEnvironment(file_name="../Reacher_Linux_1/Reacher.x86_64", 
                       no_graphics=True)
ddpg_continuous(game='Reacher-v2', 
                run=0, 
                env=env,
                remark=ddpg_continuous.__name__)