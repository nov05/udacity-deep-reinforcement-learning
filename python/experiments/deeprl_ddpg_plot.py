import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)
from deeprl import *



def plot_ddpg():
    plotter = Plotter()
    games = [
        'Reacher-v2',
    ]
    patterns = [
        'remark_ddpg',
    ]
    labels = [
        'DDPG',
    ]
    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg='mean',
                       downsample=0,  ## no downsample
                       labels=labels,
                       right_align=False,
                    #    tag=plotter.RETURN_TRAIN,
                       tag=plotter.RETURN_TEST,
                       root='.\\data\\tf_log',
                       interpolation=0,
                       window=0,
                       )
    # plt.show()
    plt.tight_layout()
    plt.savefig('data/images/unity-reacher-v2_test.png', bbox_inches='tight')



if __name__ == '__main__':
    # try:
    #     rmdir('data\\images')
    # except:
    #     pass
    mkdir('data\\images')
    plot_ddpg()

## $ python -m experiments.deeprl_ddpg_plot          <- plot tensorflow log data (tf_log)