import pandas as pd
## local imports
from unityagents import UnityEnvironment
from deeprl import *


## run "python -m experiments.unity_multiprocessing"
if __name__ == '__main__':

    # file_name = '..\data\Reacher_Windows_x86_64_1\Reacher.exe'
    env_file_name = '..\data\Reacher_Windows_x86_64_20\Reacher.exe'
    env_fn_kwargs = {'file_name': env_file_name, 'no_graphics': True}
    task = Task('unity-Reacher-v2', env_fn_kwargs=env_fn_kwargs, single_process=False)

    scores = np.zeros(task.envs_wrapper.num_agents) 
    env_id = 0
    for i in range(10000):
        actions = [np.random.randn(task.envs_wrapper.num_agents, task.action_space.shape[0])] * task.num_envs
        _, rewards, dones, infos = task.step(actions)
        scores += rewards[env_id]
        if np.any(rewards[env_id]):
            print(pd.DataFrame([rewards[env_id], scores], index=['rewards','scores']))
        if np.any(dones[env_id]): ## if any agent finishes an episode
            print("An agent finished an episode!")
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
    task.close()