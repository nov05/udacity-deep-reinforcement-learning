## local imports
from deeprl import *


## refer to D:\github\udacity-deep-reinforcement-learning\python\deeprl_files\examples.py 
##          (run "python -m deeprl_files.examples" in terminal)


## run "python -m experiments.deeprl_ddpg_continous" in terminal
def ddpg_continuous(**kwargs): 
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.max_steps = 10 #int(1e6)  ## 1,000,000
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


## in the dir "./python", run "python -m deeprl_files.examples" in terminal
if __name__ == '__main__':
    
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed()
    ## -1 is CPU, an non-negative integer is the index of GPU
    # select_device(-1)
    select_device(0) ## GPU
    game = 'Reacher-v2'
    ddpg_continuous(game=game)