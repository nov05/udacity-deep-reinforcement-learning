from deeprl.utils.misc import rmdir

if __name__ == '__main__':

    try:
        rmdir('log')
        rmdir('tf_log')
        rmdir('data')
        rmdir('image')
    except:
        pass

## $ python -m tests2.test_rmdir