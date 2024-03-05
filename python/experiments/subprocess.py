import subprocess

## check this post. https://www.datacamp.com/tutorial/python-subprocess  

## in ".\python" dir, run "python -m experiments.subprocess"
if __name__ == '__main__':
    
    example = 1

    if example==0:
        p = subprocess.Popen(["python", "--version"], 
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, errors = p.communicate()
        print(output)
        ## output: e.g. Python 3.11.7

    elif example==1: ## powershell "ls | findstr README"
        ls_process = subprocess.Popen(["dir"], shell=True, stdout=subprocess.PIPE, text=True)
        # output, error = ls_process.communicate()
        # print(output)
        # print(error)
        findstr_process = subprocess.Popen(["findstr", "README"], shell=True, 
            stdin=ls_process.stdout, stdout=subprocess.PIPE, text=True)
        output, error = findstr_process.communicate()
        print(output)
        print(error)
        ## output: e.g. 02/10/2024  08:49 PM               276 README.md