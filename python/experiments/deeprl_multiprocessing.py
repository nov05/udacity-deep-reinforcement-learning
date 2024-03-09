from deeprl import *


## in the "./python" dir, run "python -m experiments.deeprl_multiprocessing"
if __name__ == '__main__':
    
    task = Task('Hopper-v2', num_envs=10, single_process=False) ## multiprocessing doesn't work in Windows
    state = task.reset()
    for _ in range(100):
        actions = [np.random.rand(task.action_space.shape[0])] * task.num_envs
        _, _, dones, _ = task.step(actions)
        if np.any(dones):
            print(dones)
    task.close()

# $ python -m experiments.deeprl_multiprocessing