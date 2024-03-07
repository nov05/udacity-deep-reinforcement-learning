## local imports
from deeprl import *


## refer to D:\github\udacity-deep-reinforcement-learning\python\deeprl_files\examples.py 
##          (run "python -m deeprl_files.examples")
## run "python -m experiments.deeprl_ddpg_continous" in terminal

def ddpg_continuous(**kwargs): 
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game, env_fn_kwargs=config.env_fn_kwargs, single_process=True)
    config.eval_env = config.task_fn()
    config.max_steps = 10 #int(1e6)  ## 1,000,000
    config.eval_interval = int(1e4)
    config.eval_episodes = 20
    config.save_interval = int(1e4)

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


## in the dir "./python", run "python -m experiments.deeprl_ddpg_continous" in terminal
if __name__ == '__main__':
    
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed()
    ## -1 is CPU, an non-negative integer is the index of GPU
    # select_device(-1)
    select_device(0) ## GPU

    # env_file_name = '..\data\Reacher_Windows_x86_64_1\Reacher.exe'
    env_file_name = '..\data\Reacher_Windows_x86_64_20\Reacher.exe'
    env_fn_kwargs = {'file_name': env_file_name, 'worker_id':1, 'no_graphics': True}
    # task = Task('unity-Reacher-v2', env_fn_kwargs=env_fn_kwargs, single_process=True)
    ddpg_continuous(game='unity-Reacher-v2', run=0,
                    env_fn_kwargs=env_fn_kwargs,
                    remark=ddpg_continuous.__name__)
    
## unsuccessful