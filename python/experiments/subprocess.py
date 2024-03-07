import subprocess

## check this post. https://www.datacamp.com/tutorial/python-subprocess  


def test1():
    p = subprocess.Popen(["python", "--version"], 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, errors = p.communicate()
    if output: print(output.strip())
    ## output: e.g. Python 3.11.7


def test2(): ## powershell "ls | findstr README"
    ls_process = subprocess.Popen(["dir"], shell=True, stdout=subprocess.PIPE, text=True)
    # output, error = ls_process.communicate()
    # print(output)
    # print(error)
    findstr_process = subprocess.Popen(["findstr", "README"], shell=True, 
        stdin=ls_process.stdout, stdout=subprocess.PIPE, text=True)
    output, error = findstr_process.communicate()
    if output: print(output.strip())
    if error: print(error.strip())
    ## output: e.g. 02/10/2024  08:49 PM               276 README.md



if __name__ == '__main__':
    test1()
    test2()

## in ".\python" dir, run "python -m experiments.subprocess"