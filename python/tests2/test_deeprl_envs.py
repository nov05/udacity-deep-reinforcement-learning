#######################################################################
##    
##  test functions in deeprl.component.envs
##    
#######################################################################
## in the dir "./python"
## run "python -m tests2.test_deeprl_envs" in terminal
    
import pandas as pd

## local imports
from deeprl import *
from unityagents import *
from unitytrainers import *



def test1():
    ## deeprl's multiprocessing doesn't work in Windows?
    ## Hopper is a gym game.
    task = Task('Hopper-v2', num_envs=num_envs, single_process=single_process) 
    _ = task.reset() ## return state
    for _ in range(max_steps):
        actions = [np.random.rand(task.action_space.shape[0])] * task.num_envs
        _, _, dones, _ = task.step(actions)
        if np.any(dones):
            print(dones)
    task.close()

def test2():
    env = UnityEnvironment(file_name=env_file_name, seed=seed, no_graphics=False)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment 
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    _ = env_info.vector_observations                       # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    for _ in range(max_steps):
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        _ = env_info.rewards                               # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        if np.any(dones):                                  # exit loop if episode finished
            print("An agent finished an episode!")
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
    env.close()

def test3():
    env = UnityEnvironment(file_name=env_file_name, seed=seed, no_graphics=False)
    ## add "unity-" in front of the game name, or it will be considered as the gym version
    task = Task('unity-reacher-v2', envs=[env], single_process=True) 
    scores = np.zeros(task.envs_wrapper.num_agents) 
    env_id = 0
    for _ in range(max_steps):
        actions = [np.random.randn(task.envs_wrapper.num_agents, task.action_space.shape[0])] * task.num_envs
        _, rewards, dones, infos = task.step(actions)
        scores += rewards[env_id]
        if np.any(rewards[env_id]):
            print(pd.DataFrame([rewards[env_id], scores], index=['rewards','scores']))
            # print("max reached:", infos[env_id].max_reached)
        if np.any(dones[env_id]): ## if any agent finishes an episode
            print("An agent finished an episode!")
            break
    print(f"🟢 Env {env_id}, total score (averaged over agents) this episode: {np.mean(scores)}")
    task.close()

def test4():
    env_fn_kwargs = {'file_name': env_file_name, 'no_graphics': False}
    task = Task('unity-Reacher-v2', num_envs=num_envs, env_fn_kwargs=env_fn_kwargs, single_process=single_process)
    scores = np.zeros(task.envs_wrapper.num_agents) 
    env_id = 1
    for _ in range(max_steps):
        actions = [np.random.randn(task.envs_wrapper.num_agents, task.action_space.shape[0])] * task.num_envs
        _, rewards, dones, _ = task.step(actions) ## next_states, rewards, dones, infos
        scores += rewards[env_id]
        if np.any(rewards[env_id]):
            print(pd.DataFrame([rewards[env_id], scores], index=['rewards','scores']))
            # print("max reached:", infos[env_id].max_reached)
        if np.any(dones[env_id]): ## if any agent finishes an episode
            print("An agent finished an episode!")
            break
    print(f'🟢 Env {env_id}, total score (averaged over agents) this episode: {np.mean(scores)}')
    task.close()

def test5():
    # env = UnityEnvironment(file_name=env_file_name, no_graphics=True)
    env = UnityEnvironment(file_name=env_file_name, seed=seed, 
                                no_graphics=True, num_envs=2)
    task = Task('unity-Reacher-v2', envs=[env], single_process=True)
    scores = np.zeros(task.envs_wrapper.num_agents) 
    env_id = 0 ## aka. proc_id for UnityMultiEnvironment
    for _ in range(max_steps):
        actions = [np.random.randn(task.envs_wrapper.num_agents, task.action_space.shape[0])] * task.num_envs
        _, rewards, dones, _ = task.step(actions) ## next_states, rewards, dones, infos
        scores += rewards[env_id]
        if np.any(rewards[env_id]):
            print(pd.DataFrame([rewards[env_id], scores], index=['rewards','scores']))
        if np.any(dones[env_id]): ## if any agent finishes an episode
            print("An agent finished an episode!")
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
    task.close()



if __name__ == '__main__':

    # env_file_name = '..\data\Reacher_Windows_x86_64_1\Reacher.exe'
    env_file_name = '..\data\Reacher_Windows_x86_64_20\Reacher.exe'
    max_steps = 100 ## banana:1000, reacher:10000
    num_envs = 2
    env_ids = None ## None or a list
    train_mode = False 
    no_graphics = False
    seed = random.randint(-2147483648, 2147483647) ## unity env seed
    single_process = True

    # test1() ## gym fn, deeprl
    # test2() ## unity env (no deeprl)
    # test3() ## unity env, deeprl, single process
    test4() ## unity env_fn, deeprl, single process, 2 envs
    # test5() ## unity multi envs, deeprl

## $ python -m tests2.test_deeprl_envs 