# import multiprocessing as mp
import multiprocess as mp
from baselines.common.vec_env.vec_env import CloudpickleWrapper
from baselines.common.vec_env.subproc_vec_env import worker
from unityagents import UnityEnvironment

def unity_make():
    return lambda: UnityEnvironment(file_name=file_name, no_graphics=True) 

if __name__ == '__main__':

    file_name = '..\data\Reacher_Windows_x86_64_20\Reacher.exe'
    env_fn = unity_make()

    context='spawn'
    ctx = mp.get_context(context)
    remote, work_remote = ctx.Pipe()
    p = ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
    p.daemon = True  # if the main process crashes, we should not cause things to hang
    p.start()
    print(f"🟢 {p.name} has started.")
    p.join()  ## Ctrl+C to terminate the process

## $ python -m experiments.unity_cloudpickle