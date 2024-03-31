## local imports
from deeprl import *
from deeprl.utils.misc import rmdir, run_episodes, eval_episodes



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
                           num_envs=config.num_workers_eval,
                           env_fn_kwargs=config.env_fn_kwargs_eval, 
                           train_mode=False,
                           single_process=False)
    config.by_episode = True
    config.max_episodes = 320
    config.eval_episodes = num_eval_episodes  ## eval n episodes per interval
    # config.eval_after_episodes = 280
    config.eval_episode_interval = 10
    config.save_after_episodes = 280 ## save model
    config.save_episode_interval = 10 ## save model

    config.network_fn = lambda: DeterministicActorCriticNet(
        config.state_dim, 
        config.action_dim,
        actor_body=FCBody(config.state_dim, (128, 128), gate=F.relu),
        critic_body=FCBody(config.state_dim + config.action_dim, (128, 128), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4))

    config.min_memory_size = int(1e6)
    config.replay_fn = lambda: UniformReplay(memory_size=config.min_memory_size, 
                                             batch_size=config.mini_batch_size)
    config.discount = 0.99  ## λ lambda, Q-value discount rate
    config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
        size=(config.action_dim,), std=LinearSchedule(0.2))
    config.warm_up = int(1e4)
    config.target_network_mix = 1e-3  ## τ soft update rate=0.1%, trg = trg*(1-τ) + src*τ

    if is_training:
        # run_steps(DDPGAgent(config))
        run_episodes(DDPGAgent(config))
    else:
        config.save_filename = save_filename
        eval_episodes(DDPGAgent(config))



if __name__ == '__main__':

    is_training = False
    save_filename = r'data\DDPGAgent-unity-Reacher-v2-remark_ddpg_continuous-run-0-310'

    env_file_name = '..\data\Reacher_Windows_x86_64_1\Reacher.exe'
    # env_file_name = '..\data\Reacher_Windows_x86_64_20\Reacher.exe'
    set_one_thread() 
    random_seed()

    if is_training:
        # $ python -m tests2.test_rmdir
        ## remove all log and saved files
        try:
            rmdir('data\\log')
            rmdir('data\\tf_log')
            rmdir('data')
        except:
            pass
        mkdir('data\\log')
        mkdir('data\\tf_log')
        mkdir('data') ## trained models
        select_device(0) ## GPU, an non-negative integer is the index of GPU
        num_envs = 1
        num_envs_eval = 10
        eval_no_graphics = True
        num_eval_episodes = 1
    else:
        select_device(-1)  ## CPU
        num_envs = 1
        num_envs_eval = 1
        eval_no_graphics = False
        num_eval_episodes = 1

    env_fn_kwargs = {'file_name': env_file_name, 'no_graphics': True}
    env_fn_kwargs_eval = {'file_name': env_file_name, 'no_graphics': eval_no_graphics, 
                          'base_port':5005+num_envs}
    ddpg_continuous(game='unity-Reacher-v2', 
                    run=0,
                    env_fn_kwargs=env_fn_kwargs,
                    env_fn_kwargs_eval=env_fn_kwargs_eval,
                    num_workers=num_envs,
                    num_workers_eval=num_envs_eval,
                    remark=ddpg_continuous.__name__)
    


## $ python -m experiments.deeprl_ddpg_continuous    <- run this file, train or eval
## $ python -m experiments.deeprl_ddpg_plot          <- plot tensorflow log data (tf_log)
## $ python -m deeprl_files.examples
## $ python -m tests2.test_deeprl_envs
## $ python -m tests2.test_rmdir   <- delete logs, models, plots. run with caution

## example network architecture for "unity-Reacher-v2"
'''
DeterministicActorCriticNet(
  (phi_body): DummyBody()
  (actor_body): FCBody(
    (layers): ModuleList(
      (0): Linear(in_features=33, out_features=256, bias=True)
      (1): Linear(in_features=256, out_features=128, bias=True)
    )
  )
  (critic_body): FCBody(
    (layers): ModuleList(
      (0): Linear(in_features=37, out_features=256, bias=True)
      (1): Linear(in_features=256, out_features=128, bias=True)
    )
  )
  (fc_action): Linear(in_features=128, out_features=4, bias=True)
  (fc_critic): Linear(in_features=128, out_features=1, bias=True)
)
'''