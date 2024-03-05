import gym

## "python -m experiments.gym_custome_env"
if __name__ == '__main__':
    env_dict = gym.envs.registration.registry.env_specs.copy()
    for item in env_dict.items():
        print(item)