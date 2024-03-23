## local imports
from deeprl import *
from deeprl.utils.misc import rmdir


## refer to D:\github\udacity-deep-reinforcement-learning\python\deeprl_files\examples.py 
##          $ python -m deeprl_files.examples
## refer to D:\github\udacity-deep-reinforcement-learning\python\tests2\test_deeprl_envs.py
##          $ python -m tests2.test_deeprl_envs


def ddpg_continuous(**kwargs): 
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda:Task(config.game, 
                                 num_envs=config.num_workers,
                                 env_fn_kwargs=config.env_fn_kwargs, 
                                 train_mode=True,
                                 single_process=False)
    config.eval_env = Task(config.game, 
                           num_envs=1,
                           env_fn_kwargs=env_fn_kwargs_eval, 
                           train_mode=False,
                           single_process=True)
    config.max_steps = 20000 #int(1e6)  ## 1,000,000
    config.eval_interval = int(1e3)
    config.eval_episodes = 1
    config.save_interval = int(1e3)

    config.network_fn = lambda: DeterministicActorCriticNet(
        config.state_dim, 
        config.action_dim,
        actor_body=FCBody(config.state_dim, (400, 300), gate=F.relu),
        critic_body=FCBody(config.state_dim + config.action_dim, (400, 300), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    config.replay_fn = lambda: UniformReplay(memory_size=int(1e6), batch_size=100)
    config.discount = 0.99
    config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
        size=(config.action_dim,), std=LinearSchedule(0.2))
    config.warm_up = int(1e3)
    config.target_network_mix = 5e-3  ## soft update rate 0.5%, trg = trg*(1-τ) + src*τ

    run_steps(DDPGAgent(config))
    # try:
    #     run_steps(DDPGAgent(config))
    # except Exception as e:
    #     print(e)
    #     if config.eval_env is not None: config.eval_env.close()


if __name__ == '__main__':

    ## $ python -m tests2.test_rmdir
    try:
        rmdir('log')
        rmdir('tf_log')
        rmdir('data')
    except:
        pass

    mkdir('log')
    mkdir('tf_log')
    mkdir('data') ## trained models
    set_one_thread()
    random_seed()
    ## -1 is CPU, an non-negative integer is the index of GPU
    # select_device(-1)
    select_device(0) ## GPU

    num_envs = 10
    # env_file_name = '..\data\Reacher_Windows_x86_64_1\Reacher.exe'
    env_file_name = '..\data\Reacher_Windows_x86_64_20\Reacher.exe'
    env_fn_kwargs = {'file_name': env_file_name, 'no_graphics': True}
    env_fn_kwargs_eval = {'file_name': env_file_name, 'no_graphics': True, 
                          'base_port':5005+num_envs}
    # task = Task('unity-Reacher-v2', env_fn_kwargs=env_fn_kwargs, single_process=True)
    ddpg_continuous(game='unity-Reacher-v2', 
                    run=0,
                    env_fn_kwargs=env_fn_kwargs,
                    num_workers=num_envs,
                    remark=ddpg_continuous.__name__)
    
## $ python -m experiments.deeprl_ddpg_continous
## $ python -m tests2.test_rmdir   <- delete logs, models, plots. run with cauthion
## $ python -m deeprl_files.examples
## $ python -m tests2.test_deeprl_envs

## network architecture for "unity-Reacher-v2"
'''
DeterministicActorCriticNet(
  (phi_body): DummyBody()
  (actor_body): FCBody(
    (layers): ModuleList(
      (0): Linear(in_features=33, out_features=400, bias=True)
      (1): Linear(in_features=400, out_features=300, bias=True)
    )
  )
  (critic_body): FCBody(
    (layers): ModuleList(
      (0): Linear(in_features=37, out_features=400, bias=True)
      (1): Linear(in_features=400, out_features=300, bias=True)
    )
  )
  (fc_action): Linear(in_features=300, out_features=4, bias=True)
  (fc_critic): Linear(in_features=300, out_features=1, bias=True)
)
'''