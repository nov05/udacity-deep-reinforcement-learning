## created by nov05 on Oct 17, 2024
## 1. find the config here or in "./deeprl/utils/config.py"
## 2. refer to ".\deeprl_files\examples.py" 
##    $ python -m deeprl_files.examples
## 3. refer to "\tests2\test_deeprl_envs.py"
##    $ python -m tests2.test_deeprl_envs



from torch import nn
import torch.nn.functional as F
import wandb
import os

## local imports
from deeprl import *
from deeprl.utils.misc import rmdir, run_episodes, eval_episodes
from deeprl.agent.MADDPG_agent import MADDPGAgent

## log training process with W&B if uncommented
# os.environ['WANDB_MODE'] = 'disabled'



def maddpg_continuous(**kwargs): 
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.by_episode = True  ## control by episode; if false, by step
    config.max_episodes = int(1e6)  ## if lucky 2000 is sufficient for Unity Tennis
    # config.by_episode = False
    # config.max_steps = int(1e6)
    config.reset_interval = 4999  ## unity tennis env has to be reset after 5000 steps

    ## train
    if config.num_workers>0:
        config.task = Task(config.game, 
                           num_envs=config.num_workers,
                           env_fn_kwargs=config.env_fn_kwargs, 
                           train_mode=True,
                           single_process=False)
    ## eval
    if config.num_workers_eval>0:
        config.eval_env = Task(config.game, 
                               num_envs=config.num_workers_eval,
                               env_fn_kwargs=config.env_fn_kwargs_eval, 
                               train_mode=False,
                               single_process=False)
    if config.num_workers<=0:
        config.task = config.eval_env  ## some functions get info from the task object
    config.eval_episodes = num_eval_episodes  ## eval n episodes per interval
    config.eval_after_episodes = 4000
    config.eval_episode_interval = 500

    ## save
    config.save_after_episodes = 4000 ## save model
    config.save_episode_interval = 500 ## save model

    ## neural network
    config.network_fn = lambda: DeterministicActorCriticNet(
        config.state_dim,    ## dummy input length, not in use in this case
        config.action_dim,   ## actor output length
        actor_body=FCBody(config.state_dim, 
                          (256,256), 
                          gate=nn.LeakyReLU, 
                        #   noisy_linear=True,
                          init_method='uniform_fan_in', 
                          batch_norm_fn=nn.BatchNorm1d,  ## after the 1st fully connected layer
                          ),
        critic_body_fn=lambda: FCBody(  ## (x, a_1, ..., a_n)
                           (config.state_dim+config.action_dim)*config.task.envs_wrapper.num_agents,  
                           (256,256), 
                           gate=nn.LeakyReLU,
                        #    noisy_linear=True, 
                           init_method='uniform_fan_in', 
                           batch_norm_fn=nn.BatchNorm1d ## after the 1st fully connected layer
                          ),
        actor_opt_fn=lambda params: torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-5),
        critic_opt_fn=lambda params: torch.optim.AdamW(params, lr=1e-4, weight_decay=3e-5),
        critic_ensemble=True,
        num_critics=2,  ## DDPG, TD3, etc.
        )
    
    ## replay settings
    config.min_memory_size = int(1e5)
    config.mini_batch_size = 256
    # config.replay_fn = lambda: UniformReplay(memory_size=config.min_memory_size, 
    #                                          batch_size=config.mini_batch_size)
    config.replay_type = config.PRIORITIZED_REPLAY
    config.replay_fn = lambda: PrioritizedReplay(memory_size=config.min_memory_size, 
                                                 batch_size=config.mini_batch_size)
    config.discount = 0.99  ## λ lambda, Q-value discount rate
    # config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
    #     size=(config.action_dim,), std=LinearSchedule(0.2))
    config.random_process_fn = lambda: GaussianProcess(
        size=(config.action_dim,), std=LinearSchedule(1))
    config.action_noise_factor = 0.15
    config.policy_noise_factor = 0.05
    config.noise_clip = (-0.2, 0.2)
    ## before it is warmed up, use random actions, do not sample from buffer or update neural networks
    config.warm_up = int(1e4) ## can't be 0, or it will create a deadloop in buffer
    config.replay_interval = 1  ## replay-policy update every n steps
    config.actor_network_update_freq = 2  ## update the actor once for every n updates to the critic
    config.target_network_mix = int(5e-3)  ## τ: soft update rate = 0.5%, trg = trg*(1-τ) + src*τ
    # config.state_normalizer = MeanStdNormalizer()  ## value bound in range [-10, 10]
    # config.reward_normalizer = RescaleNormalizer(0.1)

    config.wandb = True
    wandb.init(
        # set the wandb project where this run will be logged
        project="udacity-drlnd-matd3-unity-tennis",
        config=config
    )
    if is_training:
        # run_steps(MADDPGAgent(config))  ## log by steps
        run_episodes(MADDPGAgent(config))  ## log by episodes
    else:
        config.save_filename = save_filename
        eval_episodes(MADDPGAgent(config))
    wandb.finish()



if __name__ == '__main__':



    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--is_training', required=False, help='whether it is a training task')
    args = parser.parse_args()
    if args.is_training is None or args.is_training.lower()=='false':
        is_training = False
    elif args.is_training.lower()=='true':
        is_training = True
    else:
        raise ValueError("⚠️ Argument '--is_training' has wrong value. Enter True or False.")

    ## game file path
    env_file_name = '..\data\Tennis_Windows_x86_64\Tennis.exe'
    # env_file_name = '..\data\Soccer_Windows_x86_64\Soccer.exe'
    ## path to the saved torch model
    save_filename = '.\data\models\MADDPGAgent-unity-tennis-remark_maddpg_continuous-run-0-2200'

    set_one_thread() 
    random_seed()

    if is_training==True:
        ## remove all log and saved files
        try:
            rmdir('data') ## remove dir ..\python\data (included in the gitignored file?) 
            rmdir('wandb')  ## remove ..\python\wandb
        except:
            pass
        mkdir('data\\log')  ## human readable logs ..\python\data\log
        mkdir('data\\tf_log')  ## tensorflow logs ..\python\data\tf_log
        mkdir('data\\models')  ## trained models ..\python\data\models
        select_device(0) ## 0: GPU, an non-negative integer is the index of GPU
        num_envs = 1
        num_envs_eval = 5
        num_eval_episodes = 20
        eval_no_graphics = True
        offset = 5
    else:
        mkdir('data\\log')
        select_device(0)  ## 0: GPU, -1: CPU
        num_envs = 0
        num_envs_eval = 4
        num_eval_episodes = 150  
        eval_no_graphics = True
        ## if run train then test in the same terminal session, skip the training port range to avoid potential conflict.
        ## e.g. for training num_envs=1, num_envs_eval=2, Unity test env port offset>=3
        offset = 6

    env_fn_kwargs = {'file_name': env_file_name, 'no_graphics': True, 'base_port':5005+offset}
    env_fn_kwargs_eval = {'file_name': env_file_name, 'no_graphics': eval_no_graphics, 
                          'base_port':5005+offset+num_envs}
    maddpg_continuous(game='unity-tennis', 
                      run=0,
                      env_fn_kwargs=env_fn_kwargs,
                      env_fn_kwargs_eval=env_fn_kwargs_eval,
                      num_workers=num_envs,
                      num_workers_eval=num_envs_eval,
                      remark=maddpg_continuous.__name__)
    


## make sure you are using the "drlnd_py310" kernel.
## $ cd python                                                            <- set the current working directory
## $ python -m experiments.deeprl_maddpg_continuous --is_training True    <- run this file, train unity tennis
## $ python -m experiments.deeprl_maddpg_continuous                       <- run this file, test unity tennis
## $ python -m experiments.deeprl_maddpg_plot                             <- plot tensorflow log data (tf_log)
## $ python -m deeprl_files.examples                                      <- train mujoco reacher
## $ python -m tests2.test_deeprl_envs                                    <- test unity envs
## $ python -m tests2.test_rmdir                                          <- delete logs, models, plots. run with caution

## example network architecture for "unity-tennis"
## check the example log file
'''
2024-11-15 21:39:17,187 - root - INFO: 
DeterministicActorCriticNet(
  (phi_body): DummyBody()
  (actor_body): FCBody(
    (layers): ModuleList(
      (0): Linear(in_features=24, out_features=256, bias=True)
      (1): ReLU()
      (2): Linear(in_features=256, out_features=256, bias=True)
      (3): ReLU()
      (4): Linear(in_features=256, out_features=2, bias=True)
      (5): Tanh()
    )
  )
  (critic_bodies): ModuleList(
    (0-1): 2 x FCBody(
      (layers): ModuleList(
        (0): Linear(in_features=52, out_features=256, bias=True)
        (1): ReLU()
        (2): Linear(in_features=256, out_features=256, bias=True)
        (3): ReLU()
        (4): Linear(in_features=256, out_features=1, bias=True)
      )
    )
  )
)
'''