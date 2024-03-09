import numpy as np
import pandas as pd
import random

## local imports
from unityagents import UnityEnvironment, UnityMultiEnvironment


def test():
    brain_name = env.brain_names[0] ## there is only brain for the executables that I have
    brain = env.brains[brain_name]
    if isinstance(env, UnityEnvironment):
        print("Running an instance of UnityEnvironment, which has only one env...")
        env_id = 0
        env_info = env.reset(train_mode=train_mode)[brain_name]
    else:
        print("Running an instance of UnityMultiEnvironment, which has multiple envs...")
        env_id = random.choice(env_ids) if env_ids is not None else 0
        print(f"Displaying env {env_id} info...")
        env_info = env.reset(train_mode=train_mode)[env_id][brain_name]  # reset the environments
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    _ = env_info.vector_observations                       # get the current state (for each agent)

    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    for i in range(max_steps):
        if isinstance(env, UnityEnvironment):
            actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
            actions = np.clip(actions, -1, 1)              # all actions between -1 and 1
            # actions = np.random.randint(num_agents, action_size) ## descrete
            env_info = env.step(actions)[brain_name]           # send all actions to the environment
        else:
            actions = np.random.randn(num_envs, num_agents, action_size) # select an action (for each agent)
            actions = np.clip(actions, -1, 1)                        # all actions between -1 and 1
            # actions = np.random.randint(num_envs, num_agents, action_size) ## descrete
            if i==0: env.step_input_check(vector_actions=actions, env_ids=env_ids) ## check only once
            env_info = env.step(actions, env_ids=env_ids, input_check=False)[env_id][brain_name] # send all actions to the environment
        _ = env_info.rewards                               # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        if np.any(dones):                                  # exit loop if episode finished
            print(f"An agent in env {env_id} finished an episode!")
            break
    print(f'🟢 Total score (averaged over agents) this episode: {np.mean(scores)}')
    env.close()


if __name__ == '__main__':

    max_steps = 10000
    num_envs = 3
    env_ids = None ## None or a list
    train_mode = False 
    no_graphics = False
    seeds = [random.randint(-2147483648, 2147483647) for _ in range(num_envs)] 
    # file_name = '..\data\Reacher_Windows_x86_64_1\Reacher.exe'
    file_name = '..\data\Reacher_Windows_x86_64_20\Reacher.exe'
    # file_name = '..\data\Banana_Windows_x86_64\Banana.exe'
    # env = UnityEnvironment(file_name=file_name, no_graphics=no_graphics) ## single env
    env = UnityMultiEnvironment(file_name=file_name, seeds=seeds,
                                no_graphics=no_graphics, num_envs=num_envs)

    # brain_name = env.brain_names[0] ## there is only brain for the executables
    # brain = env.brains[brain_name]
    # env_info = env.reset(train_mode=train_mode)[0][brain_name]  # reset the environments
    # print(env._external_brain_names); env.close(); exit()## use this line to figure out internal variables

    test()

## $ python -m tests2.test_unity_multiprocessing