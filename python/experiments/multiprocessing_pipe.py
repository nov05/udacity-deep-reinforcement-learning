# example of using a duplex pipe between processes
from time import sleep
from random import random
from multiprocessing import Process
from multiprocessing import Pipe
from multiprocessing import current_process
 
# generate and send a value
def generate_send(connection, value):
    # generate value
    new_value = random()
    # block
    sleep(new_value)
    # update value
    value = value + new_value
    # report
    print(f'>sending {value}', flush=True)
    # send value
    connection.send(value)
 
# ping pong between processes
def pingpong(connection, send_first):
    print('Process Running', flush=True)
    # check if this process should seed the process
    if send_first:
        generate_send(connection, 0)
    # run until limit reached
    while True:
        # read a value
        value = connection.recv()
        # report
        print(f'>received {value}', flush=True)
        # send the value back
        generate_send(connection, value)
        # check for stop
        if value > 10:
            break
    print('Process Done', flush=True)
 
 
# entry point
if __name__ == '__main__':
    # create the pipe
    conn1, conn2 = Pipe(duplex=True)
    # create players
    player1 = Process(target=pingpong, args=(conn1,True))
    player2 = Process(target=pingpong, args=(conn2,False))
    # start players
    player1.start()
    player2.start()
    # wait for players to finish
    player1.join()
    player2.join()
    print(player1.name, player2.name)
    print(current_process().name)

## $ python -m experiments.multiprocessing_pipe