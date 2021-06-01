# MDP with Reinforcement Learning
## By Maestrocodey

https://github.com/CodeyBank/MDP-Reinforcement-Learning

The algorithms for solving the above problem are implemented in Python (3.7 and above). Libraries used for the project include:
- [NumPy]
- [Pandas]
- Mdplib
- [Mathplotlib]
Download and install these dependencies
```sh
pip install pandas
pip install mathplotlib
pip install numpy
```

## Steps to Run the program
The following steps are to be taken to run the program
-	Open Command Line Tool. In windows, this can be done from the Command Line Prompt. In Linux Based Operating systems, it can be accessed from Terminal.
-	Change directory to project directory
-	In the command line interface, type the following

```sh
python index.py arg_1 arg_2 arg_3
```

- *arg_1: First Argument must be a valid filename in the project folder or a path to a data file properly formatted and acceptable as project requirements
-  *arg_2: Gamma Value. If not passed, program runs with loaded data
- *arg_3: Exploration factor,

## Program Architecture
The package contains the main Markov’s Decision Process Library, called mdplib.py, environment.py which is class constructor for the environment and files containing the Q-learning and Value iteration algorithms.
The program sequentially implements the value iteration and Q-learning algorithms without user interference.
The Q-Learning algorithm contains two parts: which generates actions and learns the Q function from the trials, and the simulator, which generates the trials, starting from the designated starting state and generating random reactions to the agent actions.

