from deeprl.utils.misc import rmdir

if __name__ == '__main__':

    for p in ['data\\models', 'data\\log', 'data\\tf_log', 'data\\images', 'data']:
        try:
            rmdir(p)
        except Exception as e:
            print(f'⚠️ {e}')
            pass

## $ python -m tests2.test_rmdir