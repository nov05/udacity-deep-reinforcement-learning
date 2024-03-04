import numpy as np
import matplotlib.pyplot as plt
from baselines.common import plot_util as pu

## refer to https://github.com/openai/baselines/blob/master/docs/viz/viz.ipynb

## run "python -m experiments.baselines_log" in terminal
if __name__ == '__main__':
    results = pu.load_results(r'log\bak_openai-2024-03-04-07-02-07-154853') 

    r = results[0]
    plt.plot(np.cumsum(r.monitor.l), r.monitor.r)
    plt.plot(np.cumsum(r.monitor.l), pu.smooth(r.monitor.r, radius=10), )
    plt.show()

    # pu.plot_results(results)
    # pu.plot_results(results, average_group=True)
    # pu.plot_results(results, average_group=True, split_fn=lambda _: '')
    # plt.show()