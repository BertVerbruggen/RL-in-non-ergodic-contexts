##########################################
## Reproduction on model run 19/03/2024 ##
##########################################

# Agent-based Model for Betting Behavior Analysis
# This script models and visualizes the behavior of multiple agents over different scenarios.

# ------------------------- [Imports] -------------------------
import argparse
import csv
import datetime as dt
import math
import numpy as np
import os
import pandas as pd
import pickle
import random
import time
import shutil
from matplotlib.lines import Line2D
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm, pareto

"""
# ------------------------- [Argument Parsing] -------------------------
parser = argparse.ArgumentParser(description='Train a model with given hyperparameters.')
# my_parameters = ["my_Id", "Epochs", "Runs", "Discount", "learning_rate", "Epsilon", "reward1", "reward2", "reward3"]
my_parameters = ["Id","Epochs",	"Runs	Discount", "learning_rate", "Epsilon", "reward1",	"reward2", "reward3", "epsilon_strategy",	"lr_strategy","epsilon_rate","lr_rate","epsilon_min","lr_min"]

for my_parameter in my_parameters:
    if my_parameter in ["epsilon_strategy","lr_strategy"]:
        parser.add_argument(my_parameter, type=str, help=my_parameter)
    elif my_parameter in ["Id","Epochs","Runs"]:
        parser.add_argument(my_parameter, type=int, help=my_parameter)
    else:
        parser.add_argument(my_parameter, type=float, help=my_parameter)

args = parser.parse_args()
"""

# Path to your CSV file
csv_file = "/vscmnt/brussel_pixiu_home/_user_brussel/107/vsc10780/EE_RL_Experiment/hyperparameters.csv"

# Function to convert type string to actual type
def str_to_type(type_str):
    if type_str == 'int':
        return int
    elif type_str == 'float':
        return float
    elif type_str == 'str':
        return str
    else:
        raise ValueError(f"Unknown type: {type_str}")

# Read the headers and types from the CSV file
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    headers = next(reader)  # First line: headers
    types = next(reader)    # Second line: types

# Initialize the argument parser
parser = argparse.ArgumentParser(description='Dynamic argument parsing based on CSV headers and types.')

# Dynamically add arguments based on headers and types
for header, type_str in zip(headers, types):
    arg_type = str_to_type(type_str)
    parser.add_argument(f"{header}", type=arg_type, help=f"{header} parameter of type {type_str}")

parser.add_argument("OUTPUT_DIR", type=str, help="OUTPUT_DIR")

# Parse arguments
args = parser.parse_args()

my_Id = getattr(args, "Id")
output_dir = getattr(args,"OUTPUT_DIR")

print(f"Writing output to {output_dir}")
# out_file = f"/vscmnt/brussel_pixiu_home/_user_brussel/107/vsc10780/EE_RL_Experiment/logfile_{my_Id}.log"
# newdir = "/theia/scratch/brussel/107/vsc10780/RL_Experiments/Results_MOL_" + dt.datetime.now().strftime("%m%d%Y_%H%M%S")
# newdir = output_dir
newdir = os.getcwd()

#os.mkdir(newdir)
# os.chdir(newdir)
# out_file = f"{newdir}/logfile_{my_Id}.log"
out_file = f"logfile_{my_Id}.log"

# ------------------------- [Logging and Directory Setup] -------------------------
hyperparameters = {}
with open(out_file, "w") as output:
    output.write(f"Testing for id: {my_Id}\n")
    for my_parameter in headers:
        parameter_value = getattr(args, my_parameter)
        hyperparameters[my_parameter] = parameter_value
        output.write(f"In hyperparameters dictionairy stored {my_parameter}: {parameter_value}\n")
    output.write("Parameters registered\n")

def print_to_log(linein):
    with open(out_file, "a") as handle:
        handle.write(linein + "\n")

# ------------------------- [copy current model] ------------------
def copy_current_script(target_directory):
    # Get the current script file
    current_script_path = os.path.abspath(__file__)
    
    # Copy the script to the target directory
    shutil.copy(current_script_path, target_directory)
    
copy_current_script(newdir)

print_to_log("Results stored in: " + newdir)
print_to_log("Run started at: " + dt.datetime.now().strftime("%m:%d:%Y %H:%M:%S"))

start_general_clock = dt.datetime.now()
data_file = f"DataDump_{my_Id}.p"

# Hyperparameter setting:
episodes = hyperparameters["Epochs"]
reward1  = hyperparameters["reward1"]
reward2  = hyperparameters["reward2"]
reward3  = hyperparameters["reward3"]
num_runs = hyperparameters["Runs"]
gamma    = hyperparameters["Discount"]

alpha_max      = hyperparameters["learning_rate"]
epsilon_max    = hyperparameters["Epsilon"]
alpha_min      = hyperparameters["lr_min"]
epsilon_min    = hyperparameters["epsilon_min"]
epsilon_method = hyperparameters["epsilon_strategy"]
alpha_method   = hyperparameters["lr_strategy"]
alpha_rate     = hyperparameters["lr_rate"]
epsilon_rate   = hyperparameters["epsilon_rate"]

# ------------------------- [Function Definitions] -------------------------

"""
Agent-based Model for Betting Behavior Analysis
This script models and visualizes the behavior of multiple agents over different scenarios.
"""
def initialize_q_values(method="zeros", num_actions=2, upper_bound=None):
    """
    Initialize the Q-values based on the provided method.

    Parameters:
    - method (str): The method to use for initialization. Options are "zeros", "random", "optimistic".
    - num_actions (int): The number of actions available in the environment.
    - upper_bound (float, optional): The maximum possible reward, required if using optimistic initialization.

    Returns:
    - np.array: Initialized Q-values.
    """

    if method == "zeros":
        return np.zeros(num_actions)

    elif method == "random":
        # Small random values close to zero
        return np.random.rand(num_actions) * 0.01

    elif method == "optimistic":
        if upper_bound is None:
            raise ValueError("For optimistic initialization, an upper bound on reward is required.")
        # Optimistic initialization slightly above the upper bound
        return np.full(num_actions, upper_bound + 0.01)

    else:
        raise ValueError(f"Unknown initialization method: {method}")

def write_parameters_to_csv(filename, params):
    """
    Write parameters to a CSV file.

    Parameters:
    - filename (str): The name of the CSV file.
    - params (dict): A dictionary containing the parameters to be written.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Parameter", "Value"])  # Writing headers

        for key, value in params.items():
            writer.writerow([key, value])

class DecayStrategy:
    def __init__(self, method="constant", initial_value=1.0, decay_rate=0.995, min_value=0.01):
        """
        Initialize the decay strategy.

        Parameters:
        - method (str): The strategy to use. Options are "constant", "decaying".
        - initial_value (float): Initial value for the strategy.
        - decay_rate (float): Decay rate for the strategy.
        - min_value (float): Minimum value for the strategy.
        """
        self.method = method
        self.value = initial_value
        self.decay_rate = decay_rate
        self.min_value = min_value

    def update_value(self):
        if self.method == 'constant':
            return self.value

        elif self.method == "decaying":
            self.value = max(self.value * self.decay_rate, self.min_value)
            return self.value

        else:
            raise ValueError(f"Unknown strategy method: {self.method}")

class EpsilonStrategy:
    def __init__(self, method="decaying", initial_epsilon=0.5, decay_rate=0.995, min_epsilon=0.01):
        """
        Initialize the e-strategy.

        Parameters:
        - method (str): The strategy to use. Options are "decaying", "softmax", "ucb".
        - initial_epsilon (float): Initial value of e for the "decaying" strategy.
        - decay_rate (float): Decay rate for e in the "decaying" strategy.
        - min_epsilon (float): Minimum value of e in the "decaying" strategy.
        """
        self.method = method
        self.epsilon = initial_epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon

    def update_epsilon(self, q_values=None, action_counts=None):
        """
        Update the e value based on the chosen strategy.

        Parameters:
        - q_values (np.array): Current Q-values for the state, used for "softmax" and "ucb".
        - action_counts (np.array): Number of times each action has been chosen, used for "ucb".

        Returns:
        - float: The new e value.
        """
        if self.method == 'constant':
            self.epsilon = self.epsilon

        elif self.method == "decaying":
            self.epsilon = max(self.epsilon * self.decay_rate, self.min_epsilon)

        elif self.method == "softmax":
            # The actual softmax operation to get action probabilities.
            # In this case, we are not returning an epsilon value but using softmax values for exploration.
            # Note: You'll need to adjust action selection logic in the main code for this.
            probs = np.exp(q_values) / np.sum(np.exp(q_values))
            return probs

        elif self.method == "ucb":
            # Upper Confidence Bound calculation
            # Note: You'll need to adjust action selection logic in the main code for this.
            total_actions = np.sum(action_counts)
            confidence_bounds = np.sqrt((2 * np.log(total_actions)) / np.array(action_counts))
            ucb_values = q_values + confidence_bounds
            return ucb_values

        else:
            raise ValueError(f"Unknown e-strategy method: {self.method}")

        return self.epsilon

    def select_action(self, q_values, action_counts):
        """
        Select an action based on the current Q-values and exploration strategy.

        Parameters:
        - q_values (np.array): Current Q-values for the state.
        - action_counts (np.array): Number of times each action has been chosen.

        Returns:
        - int: The selected action.
        """

        if self.method == "constant":
            return np.random.choice(actions) if np.random.uniform(0, 1) < self.epsilon else np.argmax(q_values) + 1

        elif self.method == "decaying":
            # Epsilon-greedy action selection
            return np.random.choice(actions) if np.random.uniform(0, 1) < self.epsilon else np.argmax(q_values) + 1

        elif self.method == "softmax":
            # Softmax action selection
            probs = self.update_epsilon(q_values=q_values)
            return np.random.choice(actions, p=probs)

        elif self.method == "ucb":
            # UCB action selection
            ucb_values = self.update_epsilon(q_values=q_values, action_counts=action_counts)
            return np.argmax(ucb_values) + 1

        else:
            raise ValueError(f"Unknown e-strategy method: {self.method}")

# Function to train the agont on multiple rounds for a given p using Monte Carlo Q-update and TD updates -> TD(lambda)
def monte_carlo_q_learning_with_TD(p, num_decisions):
    alpha = alpha_strategy.value

    # Initialize Q-values for the single state
    Q = initialize_q_values(method="random", upper_bound=reward1)
    Q_values_over_time = []

    # Track the number of times each action is chosen
    action_counts = np.zeros(len(actions))

    for episode in range(episodes):
        wealth = W
        episode_actions = []

        # First, take actions and update wealth for all decisions
        for decision in range(num_decisions):
            action = epsilon_strategy.select_action(Q, action_counts)
            episode_actions.append(action)

            if action == 1:
                if np.random.uniform(0, 1) < p:
                    wealth *= reward2
                else:
                    wealth *= reward1
            else:  # Safe bet
                wealth *= reward3

        # Now, update Q-values starting from the last decision
        for decision in reversed(range(num_decisions)):
            action = episode_actions[decision]

            if decision == num_decisions - 1:
                Q[action - 1] += alpha * (wealth - Q[action - 1])
            else:
                # For previous steps, we use the Q-value of the action in the next state
                # next_action = episode_actions[decision + 1] ## Original model using backlog on actions
                next_action = np.argmax(Q) # Replaced original line with selection of next action as current best action
                Q[action - 1] += alpha * (gamma * Q[next_action] - Q[action - 1])

            action_counts[action - 1] += 1

        Q_values_over_time.append(Q.copy())
        alpha = alpha_strategy.update_value()

    return Q_values_over_time

# Sarsa
def train_Sarsa(p, num_decisions):
    alpha = alpha_strategy.value

    # Initialize Q-values for the single state
    Q = initialize_q_values(method="zeros", upper_bound=reward1)
    Q_values_over_time = []

    # Track the number of times each action is chosen
    action_counts = np.zeros(len(actions))

    for episode in range(episodes):
        # Initialize the state for the start of the episode
        state = 0
        wealth = W

        # Start with an initial action
        action = epsilon_strategy.select_action(Q, action_counts)

        while state < num_decisions:
            # Compute the immediate reward based on action and update wealth
            wealth *= reward2 if (action == 1 and np.random.uniform(0, 1) < p) else reward1 if action == 1 else reward3
            reward = wealth # Define the immediate reward
            next_state = state + 1

            # Select next action based on next state
            next_action = None
            if next_state < num_decisions:
                next_action = epsilon_strategy.select_action(Q, action_counts)

            # SARSA update rule
            if next_state == num_decisions:
                # Last state update
                Q[action - 1] += alpha * (reward - Q[action - 1])
            else:
                Q[action - 1] += alpha * (reward + gamma * Q[next_action-1] - Q[action - 1])

            action_counts[action - 1] += 1

            # Prepare for next iteration
            state = next_state
            action = next_action

        Q_values_over_time.append(Q.copy())
        alpha = alpha_strategy.update_value()

    return Q_values_over_time

# Q-Learning:
def train_QLearning(p, num_decisions):
    alpha = alpha_strategy.value

    # Initialize Q-values for the single state
    Q = initialize_q_values(method="zeros", upper_bound=reward1)
    Q_values_over_time = []

    # Track the number of times each action is chosen
    action_counts = np.zeros(len(actions))

    for episode in range(episodes):
        # Initialize the state for the start of the episode
        state = 0
        wealth = W
        episode_actions = []
        episode_rewards = []

        while state < num_decisions:
            action = epsilon_strategy.select_action(Q, action_counts)
            wealth *= reward2 if (action == 1 and np.random.uniform(0, 1) < p) else reward1 if action == 1 else reward3
            episode_actions.append(action)
            episode_rewards.append(wealth)

            # Transition to the next state
            state = state + 1

        # Now, update Q-values starting from the last decision
        for decision in reversed(range(num_decisions)):
            action = episode_actions[decision]
            reward = episode_rewards[decision]

            if decision == num_decisions - 1:
                Q[action - 1] += alpha * (reward - Q[action - 1])
            else:
                Q[action - 1] += alpha * (reward + gamma * np.max(Q) - Q[action -1])
                action_counts[action - 1] += 1

        Q_values_over_time.append(Q.copy())
        alpha = alpha_strategy.update_value()

    return Q_values_over_time

# Monte Carlo
def train_MonteCarlo(p, num_decisions, update_frequency=1):
    alpha = alpha_strategy.value

    # Initialize Q-values and Returns
    Q = initialize_q_values(method="zeros", upper_bound=reward1)
    Returns = {action: [] for action in actions}
    Q_values_over_time = []

    action_counts = np.zeros(len(actions))

    for episode in range(episodes):
        step = 0
        wealth = W
        episode_memory = []  # Store (state, action, reward)

        while step < num_decisions:
            action = epsilon_strategy.select_action(Q, action_counts)
            wealth *= reward2 if (action == 1 and np.random.uniform(0, 1) < p) else reward1 if action == 1 else reward3
            episode_memory.append((action, wealth))
            step = step + 1

        # Process the episode
        for action, reward in episode_memory:
            Returns[action].append(reward)

        # Update Q-values after accumulating enough data
        if episode % update_frequency == 0:  # update_frequency decides how often to update Q-values
            for action in actions:
                if Returns[action]:
                    average_return = sum(Returns[action]) / len(Returns[action])
                    Q[action - 1] += alpha * (average_return - Q[action - 1])

        Q_values_over_time.append(Q.copy())
        alpha = alpha_strategy.update_value()

    return Q_values_over_time


# Function to train the agent for a given p
def train_agent(p):
    alpha = alpha_strategy.value

    Q = initialize_q_values(method="random", upper_bound=reward1)

    # List to store Q-values for each episode
    Q_values_over_time = []

    # Track the number of times each action is chosen
    action_counts = np.zeros(len(actions))

    for episode in range(episodes):
        action = epsilon_strategy.select_action(Q, action_counts)
        reward = reward2 if (action == 1 and np.random.uniform(0, 1) < p) else reward1 if action == 1 else reward3

        # Scale reward by starting wealth
        reward *= W

        # Current Q-value
        current_q_value = Q[action-1]

        # Max Q-value for the "next state" (which is actually the same state in this problem)
        max_next_q_value = np.max(Q)

        # Q-learning update rule
        Q[action-1] = current_q_value + alpha * (reward + gamma * max_next_q_value - current_q_value)

        # Update action counts
        action_counts[action-1] += 1

        # Store Q-values for this episode
        Q_values_over_time.append(Q.copy())

        # Update learning rate at the end of the episode
        alpha = alpha_strategy.update_value()

    return Q_values_over_time

def compute_critical_p(p_values, mean_probabilities_bet2):
    """Compute the critical p value where agent is most indifferent between, closest to,  the two bets."""
    differences = np.abs(np.array(mean_probabilities_bet2) - 0.5)
    critical_index = np.argmin(differences)
    return p_values[critical_index]

def softmax(Q):
    """ Subtract the max value from each element in the array for numerical stability"""
    shift_Q = Q - np.max(Q)
    """Compute softmax values for Q-values."""
    expQ = np.exp(shift_Q)
    return expQ / expQ.sum()

def logistic(p, L, k, p0):
    return L / (1 + np.exp(-k*(p-p0)))

def calculate_p_values_and_k(all_Q_values, p_values, episode_length):
    """Calculate the expected critical p-value and slope for a specific training length.
    Extract the Q-values for each p value at the given episode length and compute the softmax probabilities"""
    mean_probabilities_bet2 = []   # Store mean probabilities for choosing Bet 2 for each 'p'
    std_probabilities_bet2 = []    # Store standard deviations of probabilities for each 'p'

    for p in p_values:
        probs_bet2_for_p = []     # Store probabilities for choosing Bet 2 for current 'p' across all runs

        for run in all_Q_values[p]:
            Q_values_at_episode = all_Q_values[p][run][episode_length]  # Q-values at the specified episode length for each run
            ## ATTENTION: softmax function enhanced with shift for numerical stability
            prob_bet2 = softmax(Q_values_at_episode)[1]                 # Probability of choosing bet 2 using softmax on Q-values
            # prob_bet2 = np.argmax(Q_values_at_episode)
            probs_bet2_for_p.append(prob_bet2)

        # Compute mean and standard deviation for probabilities for each p
        mean_probabilities_bet2.append(np.mean(probs_bet2_for_p))
        std_probabilities_bet2.append(np.std(probs_bet2_for_p))

    # Curve fitting
    # We fit the mean probabilities to the logistic function
    try:
        params, covariance = curve_fit(logistic, p_values, mean_probabilities_bet2, p0=[1, 1, 0.5])
        L, k, p0 = params
        p0_variance = covariance[2,2]
        std_p0=np.sqrt(p0_variance)
        success = 1
    except:
        L, k, p0 = [1,1,0]
        std_p0 = 0
        success = 0

    return success, L, k, p0, std_p0, mean_probabilities_bet2, std_probabilities_bet2

def bet_1(W, p, r1, r2, N):
    """Simulate Bet 1 for N rounds."""
    wealth = W
    for _ in range(N):
        if random.random() < p:
            wealth *= r1
        else:
            wealth *= r2
    return wealth

def bet_2(W, r3, N):
    """Calculate wealth for Bet 2 after N rounds."""
    return W * (r3 ** N)

def decay_rate(a, b):
    return 1-a/b
    
# ------------------------- [Main Training Loop] -------------------------
# Define parameters
W = 10             # starting wealth

#epsilon_method = "decaying"
#epsilon_start = 0.5
#epsilon_end = 0.05
# epsilon_rate = 0.995
epsilon_rate = decay_rate(5, episodes)

#alpha_method = "constant"
#alpha_start = 0.1
#alpha_end = 0.01
alpha_rate = 0.995
# alpha_rate = decay_rate(8, episodes)

actions = [1, 2]
# epsilon_strategy = EpsilonStrategy(method=epsilon_method, initial_epsilon=epsilon_start, decay_rate = epsilon_rate, min_epsilon=epsilon_end)
# alpha_strategy = DecayStrategy(method=alpha_method, initial_value=alpha_start, decay_rate=alpha_rate, min_value=alpha_end)
epsilon_strategy = EpsilonStrategy(method=epsilon_method, initial_epsilon=epsilon_max, decay_rate = epsilon_rate, min_epsilon=epsilon_min)
alpha_strategy = DecayStrategy(method=alpha_method, initial_value=alpha_max, decay_rate=alpha_rate, min_value=alpha_min)

p_values = np.arange(0, 1.05, 0.025)
all_Q_values_single_round = {}  # Store results for single-round training
all_Q_values_multiple_rounds = {}  # Store results for multiple-rounds training

# Number of decisions to be made during training
# my_decisions = [1,2,4,6,8,10,12,14,16,18,20]
my_decisions = [1,2,5,10,15,20,25,30,50,100]

# Write main parameters to csv file
parameters = {
    "reward1": reward1,
    "reward2": reward2,
    "reward3": reward3,
    "episodes": episodes,
    "runs": num_runs,
    "discount": gamma,
    "epsilon_method": epsilon_strategy.method,
    "epsilon_initial_value": epsilon_strategy.epsilon,
    "epsilon_decay_rate": epsilon_strategy.decay_rate,
    "epsilon_min_value": epsilon_strategy.min_epsilon,
    "alpha_method": alpha_strategy.method,
    "alpha_initial_value": alpha_strategy.value,
    "alpha_decay_rate": alpha_strategy.decay_rate,
    "alpha_min_value": alpha_strategy.min_value
}

write_parameters_to_csv(f"parameters_{my_Id}.csv", parameters)

print_to_log(f"New model training started, parameters written to parameters_{my_Id}.csv\n")


## Model selection: ##
starttime = dt.datetime.now()
print_to_log(f"Project initialisation started at {dt.datetime.now()}")
models = ["Q_Learning","Sarsa","MC","II"]
model_results = {}

## Model parameters:
mc_update_frequency = 1

#------------------------- [Model Trainings] ------------------------------
# Run single-round training for comparison (outside the num_decisions loop)
all_Q_values_single_round = {}
print_to_log(f"Training the single-round model started at {dt.datetime.now()}")
for p in p_values:
    all_Q_values_single_round[p] = {}
    for run in range(num_runs):
        #epsilon_strategy = EpsilonStrategy(method=epsilon_method, initial_epsilon=epsilon_start, decay_rate = epsilon_rate, min_epsilon=epsilon_end)
        #alpha_strategy = DecayStrategy(method=alpha_method, initial_value=alpha_start, decay_rate=alpha_rate, min_value=alpha_end)
        epsilon_strategy = EpsilonStrategy(method=epsilon_method, initial_epsilon=epsilon_max, decay_rate = epsilon_rate, min_epsilon=epsilon_min)
        alpha_strategy = DecayStrategy(method=alpha_method, initial_value=alpha_max, decay_rate=alpha_rate, min_value=alpha_min)


        Q_values_over_time_single_round = train_agent(p)  # Single-round training

        all_Q_values_single_round[p][run] = Q_values_over_time_single_round

print_to_log(f"Finished training single-round model at {dt.datetime.now()}")
model_results["SingleRound"] = all_Q_values_single_round

# ------------------------- [Visualization] -------------------------

# Calculate the critical p-value and k using the function for the final episode
success, L_value, k_value, p0_value, p0_std, mean_probabilities, std_probabilities = calculate_p_values_and_k(all_Q_values_single_round, p_values, -1)

# Calculate the critical p value using a helper function
practical_p_critical = compute_critical_p(p_values, mean_probabilities)

# Theoretical critical p-value based on reward structure
theoretical_p_critical = (reward3 - reward1) / (reward2 - reward1)
theoretical_k = -(reward1 - reward2)
time_average_growth = (math.log(reward3) - math.log(reward1))/(math.log(reward2) - math.log(reward1))

# Predicted values from the logistic function using the returned k-value
fine_p_values = np.linspace(0,1,1000)
if success:
    predicted_probs = logistic(fine_p_values, L_value, k_value, p0_value)
# predicted_probs = logistic(np.array(p_values), L_value, k_value, p0_value)

# Plot
plt.figure(figsize=(10,6))
plt.errorbar(p_values, mean_probabilities, yerr=std_probabilities, fmt='o', label="Average Probabilities of Agents")
if success:
    plt.plot(fine_p_values, predicted_probs, '-', label="Fitted Logistic Curve")
else:
    plt.plot(0,0,"-",label="No logisitic fit possible")
    
plt.axhline(0.5, color='red', linestyle='--')
plt.axvline(p0_value, color='green', linestyle='--', label=f'Fitted $p_{0}$: {p0_value:.2f}') #, k: {-(k_value/4):.2f}')
plt.axvline(theoretical_p_critical, color='blue', linestyle='--', label=f'Theoretical $p_{0}$: {theoretical_p_critical:.2f}') #, k: {theoretical_k:.2f}')
# plt.axvline(practical_p_critical, color='black', linestyle='--', label=f'Calculated Critical p: {practical_p_critical:.2f}')
plt.xlabel('$\mathbb{P}$(fail risk bet)')
plt.ylabel('$\mathbb{P}$(safe)')
# plt.title('Average Probability of Choosing Bet 2 as a function of p')
plt.legend(fontsize=8)
plt.grid(True)

# plt.savefig(f"ID{my_Id}_SingleRoundModel_Policy.png",dpi = 100, bbox_inches="tight")
plt.savefig(f"ID{my_Id}_SingleRoundModel_Policy.pdf",dpi = 200, bbox_inches="tight")

# ------------------------- [Saturation Analysis] -------------------------

## Changed this number from 20 to the max statement
num_data_points = max(int(episodes/100),20)

# Calculate episodes for which we will perform the analysis
episode_increment = episodes // num_data_points
selected_episodes = [episode_increment * i for i in range(1, num_data_points)]
selected_apisodes = selected_episodes[2:].append(episodes-1)

# Initialize lists to store critical p-values and slopes for each selected episode
p0_values = []
k_values = []
L_values = []
fitted_episodes = []

for episode in selected_episodes:
    success, L, k, p0, p0_std, _, _ = calculate_p_values_and_k(all_Q_values_single_round, p_values, episode)
    if success:
        p0_values.append((p0-theoretical_p_critical)/theoretical_p_critical)
        k_values.append(k)
        L_values.append(L)
        fitted_episodes.append(episode)
    else:
        print_to_log(f"Removed episode {episode} for saturation plots due to fitting error")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left plot for critical p-values
# axes[0].plot(selected_episodes, p0_values, '-o', color='blue')
axes[0].plot(fitted_episodes, p0_values, '-o', color='blue')
# axes[0].set_title('Saturation Analysis of Critical p-value')
axes[0].set_xlabel('Training Episodes')
axes[0].set_ylabel('Critical p-value ($p_{0}$)')
axes[0].grid(True)

# Right plot for k-values
# axes[1].plot(selected_episodes, k_values, '-o', color='red')
axes[1].plot(fitted_episodes, k_values, '-o', color='red')
# axes[1].set_title('Saturation Analysis of Slope (k)')
axes[1].set_xlabel('Training Episodes')
axes[1].set_ylabel('Slope (k)')
axes[1].grid(True)

plt.tight_layout()
# plt.savefig(f"ID{my_Id}_SingleRoundModel_Saturation.png",dpi = 100, bbox_inches="tight")
plt.savefig(f"ID{my_Id}_SingleRoundModel_Saturation.pdf",dpi = 200, bbox_inches="tight", format="pdf")

# ------------------------- [Multiple-Round Model Trainings] -------------------------
print_to_log(f"Models training started at {dt.datetime.now()}")
for model_name in models:
    all_Q_values_multiple_rounds = {}
    # Loop through different numbers of rounds
    # for num_decisions in range(1, max_num_decisions + 1):
    for rounds in my_decisions:
        print_to_log(f"Training for {rounds} decision(s)")
        all_Q_values_multiple_rounds[rounds] = {}
        for p in p_values:
            all_Q_values_multiple_rounds[rounds][p] = {}
            for run in range(num_runs):
                #epsilon_strategy = EpsilonStrategy(method=epsilon_method, initial_epsilon=epsilon_start, decay_rate = epsilon_rate, min_epsilon=epsilon_end)
                #alpha_strategy = DecayStrategy(method=alpha_method, initial_value=alpha_start, decay_rate=alpha_rate, min_value=alpha_end)
                epsilon_strategy = EpsilonStrategy(method=epsilon_method, initial_epsilon=epsilon_max, decay_rate = epsilon_rate, min_epsilon=epsilon_min)
                alpha_strategy = DecayStrategy(method=alpha_method, initial_value=alpha_max, decay_rate=alpha_rate, min_value=alpha_min)


                if model_name == "Q_Learning":
                    raw_Q_values = train_QLearning(p, rounds)
                elif model_name == "Sarsa":
                    raw_Q_values = train_Sarsa(p, rounds)
                elif model_name == "MC":
                    raw_Q_values = train_MonteCarlo(p, rounds, mc_update_frequency)
                elif model_name == "II":
                    raw_Q_values = monte_carlo_q_learning_with_TD(p, rounds)
                else:
                    print_to_log(f"Model {model_name} not reckognized")

                all_Q_values_multiple_rounds[rounds][p][run] = raw_Q_values
    model_results[model_name] = all_Q_values_multiple_rounds
    print_to_log(f"Training {model_name} completed at {dt.datetime.now()}!")

## REMARK!!: Disables pickle-file output due to possible memory issues. Reactivate after test run 
# with open(data_file,"wb") as handle:
#     pickle.dump(model_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print_to_log(f"Finished training multiple-round models at {dt.datetime.now()}")
# ------------------------- [Visualization] -------------------------
# Multiple rounds Output
plt.figure(figsize=(10,6))
markers = ['o', 's', '^', 'v', '>', '<', 'p', '*', '+', 'x']
colors = plt.cm.viridis(np.linspace(0, 1, len(list(model_results.keys())) +1))
ls = 12 # font size axis labels
fs = 10 # font size legends
ts = 10 # font size ticks
x_ticks = list(i/10 for i in range(0,11,2))
x_ticklabels = list(f"{i/10:.1f}" for i in range(0,11,2))
y_ticks = x_ticks
y_ticklabels = x_ticklabels

time_average_growth = (math.log(reward3) - math.log(reward1))/(math.log(reward2) - math.log(reward1))

# Add single-run model as well:
success, L_value, k_value, p0_value, p0_std, mean_probabilities, std_probabilities = calculate_p_values_and_k(all_Q_values_single_round, p_values, -1)
if success:
    predicted_probs = logistic(fine_p_values, L_value, k_value, p0_value)
    plt.errorbar(p_values, mean_probabilities, yerr=std_probabilities,
                    color=colors[0],
                    fmt=markers[0],
                    label="Single-round model: $p_{0}$"+f"={p0_value:.2f}")
    plt.plot(fine_p_values, predicted_probs, '-',
                color=colors[0],
                )
else:
    print_to_log(f"Fitting problem for single-round model")
    plt.errorbar(p_values, mean_probabilities, yerr=std_probabilities,
                    color=colors[0],
                    fmt=markers[0],
                    label=f"Single-round model: Fit Failed!")
    
    
# Calculate the critical p-value and k using the function for the final episode
vision_index = 1
for model_name in models:
    success, L_value, k_value, p0_value, p0_std, mean_probabilities, std_probabilities = calculate_p_values_and_k(model_results[model_name][1], p_values, -1)
    if success:
        predicted_probs = logistic(fine_p_values, L_value, k_value, p0_value)
        plt.errorbar(p_values, mean_probabilities, yerr=std_probabilities,
                     color=colors[vision_index],
                     fmt=markers[vision_index],
                     label=f"{model_name}: "+"$\hat{p_{0}}$"+f"={p0_value:.2f}")
        plt.plot(fine_p_values, predicted_probs, '--',
                 color=colors[vision_index],
                 )
        vision_index += 1 if vision_index < len(colors)-1 else 0
    else:
        print_to_log(f"Fitting problem for model {model_name} for one-round iteration")
        plt.errorbar(p_values, mean_probabilities, yerr=std_probabilities,
                     color=colors[vision_index],
                     fmt=markers[vision_index],
                     label=f"{model_name}: Fit Failed!")
        

# Plot
plt.axhline(0.5, color='red', linestyle='--')
plt.axvline(theoretical_p_critical, color='blue', linestyle='--', label=f'Theoretical $p_{0}$: {theoretical_p_critical:.2f}') # , k: {-theoretical_k:.2f}')
plt.axvline(time_average_growth, color='orange', linestyle='-', label=f'Time growth average $g_t$: {time_average_growth:.2f}')
plt.xlabel('$\mathbb{P}$(fail risk bet)',fontsize=ls)
plt.ylabel('$\mathbb{P}$(safe)',fontsize=ls)
plt.xticks(fontsize=ts)
plt.yticks(fontsize=ts)
# plt.title(f'Model Probability for Bet 2 as a function of p')
plt.legend(fontsize=fs)
plt.grid(True)

# plt.savefig(f"ID{my_Id}_All_Models_FirstRound.png",dpi = 100, bbox_inches="tight",format="png")
plt.savefig(f"ID{my_Id}_All_Models_FirstRound.pdf",dpi = 200, bbox_inches="tight",format="pdf")

# ------------------------- [Visualization] -------------------------
# Multiple Models visualization
x = 0
y = 0
x_len = int(len(models)/2)
y_len = len(models)-int(len(models)/2)
_, axs = plt.subplots(x_len,y_len,figsize=(17.5,10))

## New addition 05/02/2024:
model_policies = {}
## end new lines 05/02/2024

for model_name in models:
    print_to_log(f"Generating output for model {model_name}")
    ## New addition 05/02/2024:
    model_policies[model_name] = {}
    ## end new lines 05/02/2024
    # Add single-run model as well:
    success, L_value, k_value, p0_value, p0_std, mean_probabilities, std_probabilities = calculate_p_values_and_k(all_Q_values_single_round, p_values, -1)
    if success:
        predicted_probs = logistic(fine_p_values, L_value, k_value, p0_value)
        axs[x][y].errorbar(p_values, mean_probabilities, yerr=std_probabilities,
                        color=colors[0],
                        fmt=markers[0],
                        label="Single-round model: $\hat{p_{0}}$"+f"={p0_value:.2f}")
        axs[x][y].plot(fine_p_values, predicted_probs, '-',
                    color=colors[0],
                    )

    else:
        print_to_log(f"Fit Failed for single-round model!")
        axs[x][y].errorbar(p_values, mean_probabilities, yerr=std_probabilities,
                        color=colors[0],
                        fmt=markers[0],
                        label=f"Single-round model: Fit Failed!")
        
    # Calculate the critical p-value and k using the function for the final episode
    vision_index = 1
    for round in my_decisions:
#        try:
#            L_value, k_value, p0_value, mean_probabilities, std_probabilities = calculate_p_values_and_k(model_results[model_name][round], p_values, -1)
#        except:
#            print_to_log(f"No function fit possible with results for model {model_name} on {round} rounds training")
#        else:
#            predicted_probs = logistic(fine_p_values, L_value, k_value, p0_value)
#            axs[x][y].plot(fine_p_values, predicted_probs, '--',
#                color=colors[vision_index],
#                #label=f"Fitted Critical rounds {round:.2f}: p={p0_value:.2f}"
#                )
#            axs[x][y].errorbar(p_values, mean_probabilities, yerr=std_probabilities,
#                    color=colors[vision_index],
#                    fmt=markers[vision_index],
#                    label=f"Average Probabilities for {round} rounds: p={p0_value:.2f}")

        success, L_value, k_value, p0_value, p0_std, mean_probabilities, std_probabilities = calculate_p_values_and_k(model_results[model_name][round], p_values, -1)
        if success:
            predicted_probs = logistic(fine_p_values, L_value, k_value, p0_value)
            axs[x][y].plot(fine_p_values, predicted_probs, '--',
                color=colors[vision_index],
                #label=f"Fitted Critical rounds {round:.2f}: p={p0_value:.2f}"
                )
            axs[x][y].errorbar(p_values, mean_probabilities, yerr=std_probabilities,
                    color=colors[vision_index],
                    fmt=markers[vision_index],
                    label=f"{round} rounds: "+"$\hat{p_{0}}$"+f"={p0_value:.2f}")
            model_policies[model_name][round] = [p0_value, p0_std]
        else:
            print_to_log(f"Fit Failed for model {model_name} on {round} training rounds")
            axs[x][y].errorbar(p_values, mean_probabilities, yerr=std_probabilities,
                    color=colors[vision_index],
                    fmt=markers[vision_index],
                    label=f"{round} rounds: Fit Failed!")

        if vision_index < min(len(markers),len(colors))-1:
            vision_index += 1
        else:
            vision_index = 1

    # Plot
    axs[x][y].axhline(0.5, color='red', linestyle='--')
    axs[x][y].axvline(theoretical_p_critical, color='blue', linestyle='--', label=f'Theoretical $p_{0}$: {theoretical_p_critical:.2f}')
    axs[x][y].axvline(time_average_growth, color='orange', linestyle='-', label=f'Time growth average $g_t$: {time_average_growth:.2f}')
    axs[x][y].set_xlabel('$\mathbb{P}$(fail risk bet)', fontsize=ls)
    axs[x][y].set_ylabel('$\mathbb{P}$(safe)', fontsize=ls)
    axs[x][y].set_xticks(x_ticks)
    axs[x][y].set_xticklabels(x_ticks,fontsize=ts)
    axs[x][y].set_yticks(y_ticks)
    axs[x][y].set_yticklabels(y_ticks,fontsize=ts)

    # axs[x][y].set_title(f'Model {model_name} results as a function of p')
    axs[x][y].legend(fontsize=fs)
    axs[x][y].grid(True)

    # Plot number:
    if y < y_len-1:
        y += 1
    else:
        y = 0
    if x < x_len and y == 0:
        x += 1

# plt.savefig(f"ID{my_Id}_All_Models_AllRounds.png",dpi = 100, bbox_inches="tight",format="png")
plt.savefig(f"ID{my_Id}_All_Models_AllRounds.pdf",dpi = 200, bbox_inches="tight",format="pdf")

print_to_log(f"Saved figure on multiple rounds ID{my_Id}_All_Models_AllRounds.png")

## New addition 05/02/2024:
plt.figure(figsize=(10,6))

markers = ['o', 's', '^', 'v', '>', '<', 'p', '*', '+', 'x']
colors = plt.cm.viridis(np.linspace(0, 1, len(models)+1))
legend_string = list()
vision_index = 1

for model_name, results in model_policies.items():
    for round, p_estimates in results.items():
        if p_estimates[1] > min(p_estimates[0], 1-p_estimates[0]):
            std_p0 = min(p_estimates[0], 1-p_estimates[0])
        else:
            std_p0 = p_estimates[1]
        # axs[x][y]
        plt.errorbar(round, p_estimates[0], yerr=std_p0,
                color=colors[vision_index],
                marker=markers[vision_index],
                # label=f"Average Probabilities for {round} rounds: p={p0_value:.2f}"
                )

    legend_string.append(Line2D([0], [0],
                                color=colors[vision_index],
                                marker = markers[vision_index],
                                label=f"{model_name} model")
                                )
    vision_index = vision_index+1 if vision_index < min(len(markers),len(colors))-1 else 1
    

# Plot
# plt.axhline(0.5, color='red', linestyle='--')
plt.axhline(time_average_growth, color='orange', linestyle='-', label=f'Time growth average $g_t$: {time_average_growth:.2f}')
plt.axhline(theoretical_p_critical, color='blue', linestyle='--', label=f'Theoretical $p_{0}$: {theoretical_p_critical:.2f}') # , k: {-theoretical_k:.2f}')
legend_string.append(Line2D([0], [0],
                                color="blue",
                                linestyle='--',
                                label=f'Theoretical $p_{0}$: {theoretical_p_critical:.2f}') #, k: {-theoretical_k:.2f}')
                                )
legend_string.append(Line2D([0], [0],
                                color="orange",
                                linestyle="-",
                                label=f'Time growth average $g_t$: {time_average_growth:.2f}')
                                )
                                
plt.xlabel("# game repetitions",fontsize=ls)
plt.ylabel("$\hat{p_{0}}$",fontsize=ls)
plt.xticks(fontsize=ts)
plt.yticks(fontsize=ts)
# plt.title("Estimated policy change")
plt.legend(handles=legend_string, frameon=True, loc='best',fontsize=fs)

# plt.savefig(f"ID{my_Id}_PolicyEvolution.png",dpi = 100, bbox_inches="tight",format="png")
plt.savefig(f"ID{my_Id}_PolicyEvolution.pdf",dpi = 200, bbox_inches="tight",format="pdf")
## end new lines 05/02/2024

# ------------------------- [Visualization No fit] -------------------------
# Multiple Models visualization
x = 0
y = 0
x_len = int(len(models)/2)
y_len = len(models)-int(len(models)/2)
_, axs = plt.subplots(x_len,y_len,figsize=(17.5,10))

markers = ['o', 's', '^', 'v', '>', '<', 'p', '*', '+', 'x']
colors = plt.cm.viridis(np.linspace(0, 1, len(my_decisions)+1))

def arithmeticAverages(q_table, p_values):
    arithmetic_values = []
    for p in p_values:
        runs = 0
        prob_bet2 = 0
        for run in q_table[p]:
            Q_values_at_episode = q_table[p][run][-1]  # Q-values at the specified episode length for each run
            prob_bet2 += softmax(Q_values_at_episode)[1]
            runs+=1

        arithmetic_values.append(prob_bet2/runs)

    return arithmetic_values

for model_name in models:
    # print(f"Generating output for model {model_name}")
    # Add single-run model as well:
    mean_probabilities = arithmeticAverages(all_Q_values_single_round, p_values)
    axs[x][y].plot(p_values, mean_probabilities,
                    color=colors[0],
                    marker=markers[0],
                    label=f"Average Probabilities for single-round model")

    # Calculate the critical p-value and k using the function for the final episode
    # for round in range(1, num_decisions):
    vision_index = 1
    for round in my_decisions:
        mean_probabilities = arithmeticAverages(model_results[model_name][round], p_values)
        axs[x][y].plot(p_values, mean_probabilities,
                color=colors[vision_index],
                marker=markers[vision_index],
                label=f"Average Probabilities for {round} rounds")

        vision_index = vision_index+1 if vision_index < min(len(markers),len(colors))-1 else 1

    # Plot
    axs[x][y].axhline(0.5, color='red', linestyle='--')
    axs[x][y].axvline(theoretical_p_critical, color='blue', linestyle='--', label=f'Theoretical $p_{0}$: {theoretical_p_critical:.2f}') # , k: {-theoretical_k:.2f}')
    # axs[x][y].axvline(time_growth_average, color='blue', linestyle='-', label=f'time growth average g: {time_growth_average:.2f}')
    axs[x][y].set_xlabel('$\mathbb{P}(fail risk bet)$')
    axs[x][y].set_ylabel('$\mathbb{P}$(safe)')
    # axs[x][y].set_title(f'Model {model_name} results as a function of p')
    axs[x][y].legend(fontsize=8)
    axs[x][y].grid(True)

    # Plot number:
    if y < y_len-1:
        y += 1
    else:
        y = 0
    if x < x_len and y == 0:
        x += 1

# plt.savefig(f"ID{my_Id}_All_Models_AllRounds_NoFit.png",dpi = 100, bbox_inches="tight",format="png")
plt.savefig(f"ID{my_Id}_All_Models_AllRounds_NoFit.pdf",dpi = 200, bbox_inches="tight",format="pdf")

print_to_log(f"Finished model training and analysis at {dt.datetime.now()}, Writing data to output file")

output_object = {}
for model_key, model_values in model_results.items():
    output_object[model_key]={}
    if model_key == 'SingleRound':
        for prob_key, prob_values in model_values.items():
            output_object[model_key][prob_key] = {}
            for run_key, run_values in prob_values.items():
                output_object[model_key][prob_key][run_key] = run_values[-100:]
    else:
        for round_key, round_values in model_values.items():
            output_object[model_key][round_key] = {}
            for prob_key, prob_values in round_values.items():
                output_object[model_key][round_key][prob_key] = {}
                for run_key, run_values in prob_values.items():
                    output_object[model_key][round_key][prob_key][run_key] = run_values[-100:]
                                    
with open(data_file,"wb") as handle:
    pickle.dump(output_object, handle, protocol=pickle.HIGHEST_PROTOCOL)