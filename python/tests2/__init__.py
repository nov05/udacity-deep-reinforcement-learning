## added by nov05, 20240305
import os, platform
if platform.system()=='Windows':
    os.add_dll_directory(r"C:/Users/guido/.mujoco/mjpro150/bin")
    os.add_dll_directory(r"C:/Users/guido/.mujoco/mujoco-py-1.50.1.68/mujoco_py")
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

## local imports
from unityagents import *
from unitytrainers import *
from deeprl import *

