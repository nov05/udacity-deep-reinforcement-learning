import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)
from deeprl import *



def plot_maddpg():
    plotter = Plotter()
    games = [
        # 'Reacher-v2',  ## mujoco
        # 'unity-reacher-v2',
        'unity-tennis'
    ]
    patterns = [
        'remark_maddpg',
    ]
    labels = [
        'MADDPG',
    ]
    for log_type in ['train', 'test']:
    # for log_type in ['train']:
        tag = plotter.RETURN_TRAIN if log_type=='train' else plotter.RETURN_TEST
        moving_average = 100 if log_type=='train' else 5
        plotter.plot_games(games=games,
                        patterns=patterns,
                        agg='mean',
                        moving_average=moving_average,
                        downsample=0,  ## no downsample
                        labels=labels,
                        right_align=False,
                        tag=tag,
                        root='.\\data\\tf_log',
                        interpolation=0,
                        window=0,
                        )
        # plt.show()
        plt.tight_layout()
        plt.savefig(f'data/images/{games[0]}_{log_type}.png', bbox_inches='tight')  ## output plots



if __name__ == '__main__':
    # try:
    #     rmdir('data\\images')
    # except:
    #     pass
    mkdir('data\\images')
    plot_maddpg()

## $ python -m experiments.deeprl_maddpg_plot     <- plot tensorflow log data (tf_log)