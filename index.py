import numpy as np
from scipy import misc
import mdplib as mdp
import matplotlib.pyplot as plt
import sys
import time

# Close all open plots if open
plt.close('all')

try:
    # get all the command line arguments
    filename = sys.argv[1] if len(sys.argv) > 1 else None
    gamma = float(sys.argv[2]) if len(sys.argv) > 2 else None
    epsilon = float(sys.argv[3]) if len(sys.argv) > 3 else None

    # create an instance of the world
    env = mdp.LoadEnvironment(filename,False)

    # Modify the parameters of the world if gamma and epsilon is given
    if gamma:
        env.Gamma = gamma
    if epsilon:
        env.Explor = epsilon

    #Set number of iterations
    # n_iterations = 200

    # # Execute value iteration algorithm
    # print("*** Value Iteration Algorithm will be executed for"
    #       " maximum of {} iterations and gamma = {} ***".format(n_iterations,env.Gamma))
    # V, A, tail, count = mdp.vi.ValueIteration(env, n_iterations)
    # print("*** Algorithm successfully executed in {} iterations. Agent found optimal Policy !".format(count))
    # mdp.showPolicyUtility(env, V, A)
    # title = "Plot of Utility vs Iterations for file: " + filename
    # fname = "Util_vs_Iter: " + filename
    # mdp.plotUtility(env, tail, title, fname)
    #
    # Execute Q-Learning algorithm
    n_iterations = 15000
    print("Executing Q learning algorithm for {} iterations... ".format(n_iterations))
    time.sleep(2)
    print("Please wait while I compute the optimal policy. This might take a while longer depending on your CPU power")
    Q, tail = mdp.ql.QLearning(env, n_iterations)
    mdp.PrintQResults(env, Q)
    title = "Q-Values vs Iterations for file: " + filename + " G = {} E = {}".format(env.Gamma,env.Explor)
    fname = "Q_vs_Iter G={},E={} W:".format(env.Gamma, env.Explor) + filename
    mdp.plotUtility(env, tail, title, fname)

except (TypeError, FileNotFoundError):
    print("First parameter must be the file name to be loaded. Gamma and Epsilon are optional"
          "\nBut must be keyed in this order"
          "\nFilename Gamma Epsilon")
