
## **👉 Unity ML-Agents `Banana Collectors` Project Submission**  

<img src="https://github.com/Nov05/pictures/blob/master/Udacity/20231221_reinforcement%20learning/p1_navigation_project_submission.gif?raw=true">

1. For this toy game, two `Deep Q-network` methods are tried out. Since the observations (states) are simple (not in pixels), convolutional layers are not in use. And the evaluation results confirm that linear layers are sufficient for solving the problem.   
	* **Double DQN**, with 3 linear layers (hidden dims: 256\*64, later tried with 64\*64)  
	* **Dueling DQN**, with 2 linear layers + 2 split linear layers (hidden dims: 64\*64)  

▪️ The Dueling DQN architecture is displayed as below. 

<table>
<tr>
<th> Dueling Architecture </th>
<th> The green module </th>
</tr>
<tr>
<td>
<img src="https://github.com/Nov05/pictures/blob/master/Udacity/20231221_reinforcement%20learning/2024-02-13%2012_08_13-1511.06581.pdf.jpg?raw=true" width=300>  
</td>
<td>
<img src="https://github.com/Nov05/pictures/blob/master/Udacity/20231221_reinforcement%20learning/2024-02-13%2012_11_04-1511.06581.pdf.jpg?raw=true" width=300>
</td>
</tr>
</table>  

▪️ Since both the advantage and the value stream propagate gradients to the last convolutional layer in the backward pass, we rescale the combined gradient entering the last convolutional layer by 1/√2. This simple heuristic mildly increases stability.

```Python
        self.layer1 = nn.Linear(state_size, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3_adv = nn.Linear(in_features=64, out_features=action_size) ## advantage
        self.layer3_val = nn.Linear(in_features=64, out_features=1) ## state value

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        adv, val = self.layer3_adv(x), self.layer3_val(x)
        return (val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), action_size)) / (2**0.5)
```  

▪️ In addition, we clip the gradients to have their norm less than or equal to 10. This clipping is not standard practice in deep RL, but common in recurrent network training (Bengio et al., 2013).

```Python 
        ## clip the gradients
        nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 10.)
        nn.utils.clip_grad_norm_(self.qnetwork_target.parameters(), 10.) 
```

2. The following picture shows the train and eval scores (rewards) for both architectures. Since it is a toy project, trained models are not formally evaluated. We can roughly see that Dueling DQN slightly performs better with **an average score of 17** vs. Double DQN 13 in 10 episodes.  

<img src="https://github.com/Nov05/pictures/blob/master/Udacity/20231221_reinforcement%20learning/p1-project-submission.jpg?raw=true" width=600>  

3. **Project artifacts:** 
	* [All the notebooks](https://gist.github.com/Nov05/4e0ff3edba96928facaff063039c7bce) (trained in Google Colab, evaluated on local machine)  
	* The project folder [`p1_navigation`](https://github.com/Nov05/udacity-deep-reinforcement-learning/tree/master/p1_navigation) (which contains checkpoints `dqn_checkpoint_2000.pth` and `dueling_dqn_checkpoint_2000.pth`)  
	* [Video recording](https://youtu.be/SwAwWLsa9f0?t=35) (which demonstrates how trained models are run on the local machine)  

---

## **👉 Logs**  
2024-02-14 Banana game project submission  
2024-02-11 Unity MLAgent [Banana env set up](https://gist.github.com/Nov05/bf63ac7e0a2d0f94a635fb3858894cca)  
2024-02-10 repo cloned  

---

[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135602-b0335606-7d12-11e8-8689-dd1cf9fa11a9.gif "Trained Agents"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"

# Deep Reinforcement Learning Nanodegree

![Trained Agents][image1]

This repository contains material related to Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.  

## Table of Contents

### Tutorials

The tutorials lead you through implementing various algorithms in reinforcement learning.  All of the code is in PyTorch (v0.4) and Python 3.

* [Dynamic Programming](https://github.com/udacity/deep-reinforcement-learning/tree/master/dynamic-programming): Implement Dynamic Programming algorithms such as Policy Evaluation, Policy Improvement, Policy Iteration, and Value Iteration. 
* [Monte Carlo](https://github.com/udacity/deep-reinforcement-learning/tree/master/monte-carlo): Implement Monte Carlo methods for prediction and control. 
* [Temporal-Difference](https://github.com/udacity/deep-reinforcement-learning/tree/master/temporal-difference): Implement Temporal-Difference methods such as Sarsa, Q-Learning, and Expected Sarsa. 
* [Discretization](https://github.com/udacity/deep-reinforcement-learning/tree/master/discretization): Learn how to discretize continuous state spaces, and solve the Mountain Car environment.
* [Tile Coding](https://github.com/udacity/deep-reinforcement-learning/tree/master/tile-coding): Implement a method for discretizing continuous state spaces that enables better generalization.
* [Deep Q-Network](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn): Explore how to use a Deep Q-Network (DQN) to navigate a space vehicle without crashing.
* [Robotics](https://github.com/dusty-nv/jetson-reinforcement): Use a C++ API to train reinforcement learning agents from virtual robotic simulation in 3D. (_External link_)
* [Hill Climbing](https://github.com/udacity/deep-reinforcement-learning/tree/master/hill-climbing): Use hill climbing with adaptive noise scaling to balance a pole on a moving cart.
* [Cross-Entropy Method](https://github.com/udacity/deep-reinforcement-learning/tree/master/cross-entropy): Use the cross-entropy method to train a car to navigate a steep hill.
* [REINFORCE](https://github.com/udacity/deep-reinforcement-learning/tree/master/reinforce): Learn how to use Monte Carlo Policy Gradients to solve a classic control task.
* **Proximal Policy Optimization**: Explore how to use Proximal Policy Optimization (PPO) to solve a classic reinforcement learning task. (_Coming soon!_)
* **Deep Deterministic Policy Gradients**: Explore how to use Deep Deterministic Policy Gradients (DDPG) with OpenAI Gym environments.
  * [Pendulum](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum): Use OpenAI Gym's Pendulum environment.
  * [BipedalWalker](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal): Use OpenAI Gym's BipedalWalker environment.
* [Finance](https://github.com/udacity/deep-reinforcement-learning/tree/master/finance): Train an agent to discover optimal trading strategies.

### Labs / Projects

The labs and projects can be found below.  All of the projects use rich simulation environments from [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents). In the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program, you will receive a review of your project.  These reviews are meant to give you personalized feedback and to tell you what can be improved in your code.

* [The Taxi Problem](https://github.com/udacity/deep-reinforcement-learning/tree/master/lab-taxi): In this lab, you will train a taxi to pick up and drop off passengers.
* [Navigation](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation): In the first project, you will train an agent to collect yellow bananas while avoiding blue bananas.
* [Continuous Control](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control): In the second project, you will train an robotic arm to reach target locations.
* [Collaboration and Competition](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet): In the third project, you will train a pair of agents to play tennis! 

### Resources

* [Cheatsheet](https://github.com/udacity/deep-reinforcement-learning/blob/master/cheatsheet): You are encouraged to use [this PDF file](https://github.com/udacity/deep-reinforcement-learning/blob/master/cheatsheet/cheatsheet.pdf) to guide your study of reinforcement learning. 

## OpenAI Gym Benchmarks

### Classic Control
- `Acrobot-v1` with [Tile Coding](https://github.com/udacity/deep-reinforcement-learning/blob/master/tile-coding/Tile_Coding_Solution.ipynb) and Q-Learning  
- `Cartpole-v0` with [Hill Climbing](https://github.com/udacity/deep-reinforcement-learning/blob/master/hill-climbing/Hill_Climbing.ipynb) | solved in 13 episodes
- `Cartpole-v0` with [REINFORCE](https://github.com/udacity/deep-reinforcement-learning/blob/master/reinforce/REINFORCE.ipynb) | solved in 691 episodes 
- `MountainCarContinuous-v0` with [Cross-Entropy Method](https://github.com/udacity/deep-reinforcement-learning/blob/master/cross-entropy/CEM.ipynb) | solved in 47 iterations
- `MountainCar-v0` with [Uniform-Grid Discretization](https://github.com/udacity/deep-reinforcement-learning/blob/master/discretization/Discretization_Solution.ipynb) and Q-Learning | solved in <50000 episodes
- `Pendulum-v0` with [Deep Deterministic Policy Gradients (DDPG)](https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/DDPG.ipynb)

### Box2d
- `BipedalWalker-v2` with [Deep Deterministic Policy Gradients (DDPG)](https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/DDPG.ipynb)
- `CarRacing-v0` with **Deep Q-Networks (DQN)** | _Coming soon!_
- `LunarLander-v2` with [Deep Q-Networks (DQN)](https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/solution/Deep_Q_Network_Solution.ipynb) | solved in 1504 episodes

### Toy Text
- `FrozenLake-v0` with [Dynamic Programming](https://github.com/udacity/deep-reinforcement-learning/blob/master/dynamic-programming/Dynamic_Programming_Solution.ipynb)
- `Blackjack-v0` with [Monte Carlo Methods](https://github.com/udacity/deep-reinforcement-learning/blob/master/monte-carlo/Monte_Carlo_Solution.ipynb)
- `CliffWalking-v0` with [Temporal-Difference Methods](https://github.com/udacity/deep-reinforcement-learning/blob/master/temporal-difference/Temporal_Difference_Solution.ipynb)

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. If running in **Windows**, ensure you have the "Build Tools for Visual Studio 2019" installed from this [site](https://visualstudio.microsoft.com/downloads/).  This [article](https://towardsdatascience.com/how-to-install-openai-gym-in-a-windows-environment-338969e24d30) may also be very helpful.  This was confirmed to work in Windows 10 Home.  

3. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
4. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.  
    ```bash
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .
    ```

5. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.    
    ```bash
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```

6. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]

## Want to learn more?

<p align="center">Come learn with us in the <a href="https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893">Deep Reinforcement Learning Nanodegree</a> program at Udacity!</p>

<p align="center"><a href="https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893">
 <img width="503" height="133" src="https://user-images.githubusercontent.com/10624937/42135812-1829637e-7d16-11e8-9aa1-88056f23f51e.png"></a>
</p>
