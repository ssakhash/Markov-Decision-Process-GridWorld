import numpy as np
import random
import os
from copy import copy, deepcopy

#Extracts data from the Input txt File, assigns it to variables and returns the data to the main() function.
def read_inputfile():
    walls = []
    terminal_states = []
    terminal_state_reward = []
    transition_probabilities = []
    for files in os.listdir():
        filename, extension = os.path.splitext(files)
        if extension == '.txt':
            input_file = files
    with open(input_file) as f:
        lines = f.readlines()
    for line in lines:
        if ":" in line:
            split_line = line.split(":")
            if split_line[0].strip() == 'size':
                interim_size = split_line[1].strip()
                size = interim_size.split(" ")
                size = [int(size[0]), int(size[1])]
            if split_line[0].strip() == 'walls':
                interim_walls = split_line[1].strip()
                interim_walls = interim_walls.split(",")
                for interim_wall in interim_walls:
                    interim_wall = interim_wall.strip()
                    interim_wall = interim_wall.split(" ")
                    walls.append([int(interim_wall[0]), int(interim_wall[1])])
            if split_line[0].strip() == 'terminal_states':
                interim_terminalstates = split_line[1].strip()
                interim_terminalstates = interim_terminalstates.split(",")
                for interim_terminalstate in interim_terminalstates:
                    interim_terminalstate = interim_terminalstate.strip()
                    interim_terminalstate = interim_terminalstate.split(" ")
                    terminal_states.append([int(interim_terminalstate[0]), int(interim_terminalstate[1])])
                    terminal_state_reward.append(int(interim_terminalstate[2]))
            if split_line[0].strip() == 'reward':
                interim_reward = split_line[1].strip()
                reward = float(interim_reward)
            if split_line[0].strip() == 'transition_probabilities':
                interim_TP = split_line[1].strip()
                interim_TP = interim_TP.split(" ")
                for interim_tp in interim_TP:
                    interim_tp = interim_tp.strip()
                    transition_probabilities.append(float(interim_tp))
            if split_line[0].strip() == 'discount_rate':
                interim_discountrate = split_line[1].strip()
                discount_rate = float(interim_discountrate)
            if split_line[0].strip() == 'epsilon':
                interim_epsilon = split_line[1].strip()
                epsilon = float(interim_epsilon)
    return size, walls, terminal_states, terminal_state_reward, reward, transition_probabilities, discount_rate, epsilon

#Used to create the Initial State based on the mentioned Size of the Matrix, and the position of walls.
def create_initialstate(size, walls):
    num_rows, num_columns = size
    initial_state = np.zeros((num_columns, num_rows))
    for wall in walls:
        wall_row, wall_column = wall
        initial_state[wall_column-1][wall_row-1] = np.nan
    return initial_state

#Returns a Reward Matrix created based on the mentioned Matrix Size, Position of Terminal States, Correspnding Rewards and Transition Rewards
def create_rewardmatrix(size, transition_reward, terminal_states, terminal_state_reward):
    num_rows, num_columns = size
    reward = np.full((num_columns, num_rows), transition_reward)
    for value in range(len(terminal_states)):
        state_rows, state_columns = terminal_states[value]
        reward[abs(state_columns - num_columns)][abs(state_rows - 1)] = terminal_state_reward[value]
    return reward

#Returns the List of Possible Actions for Each Intended Action (taking into consideration the stochasticity)
def action_predictor(row, column, intended_action, transition_probabilities):
    intended_action = str(intended_action[0])
    TP = deepcopy(transition_probabilities)
    actions = ["U", "D", "L", "R"]
    if intended_action == "U":
        possible_actions = ["U", "L", "R"]
    if intended_action == "D":
        possible_actions = ["D", "R", "L"]
    if intended_action == "L":
        possible_actions = ["L", "U", "D"]
    if intended_action == "R":
        possible_actions = ["R", "U", "D"]
    corresponsing_probabilities = [TP[0], TP[1], TP[2]]
    return possible_actions, corresponsing_probabilities

#With the Position of Current Cell and Action as Inputs, the function returns the Next Cell index(taking into consideration Borders and Wall positions)
def action_outcome(initial_state, row, column, action):
    max_row, max_column = initial_state.shape
    max_row-=1
    max_column-=1
    if action == "U":
        if row == 0:
            next_row = row
            next_column = column
            return next_row, next_column
        elif np.isnan(initial_state[row - 1][column]):
            next_row = row
            next_column = column
            return next_row, next_column
        else:
            next_row = row - 1
            next_column = column
            return next_row, next_column
    if action == "D":
        if row == max_row:
            next_row = row
            next_column = column
            return next_row, next_column
        elif np.isnan(initial_state[row + 1][column]):
            next_row = row
            next_column = column
            return next_row, next_column
        else:
            next_row = row + 1
            next_column = column
            return next_row, next_column
    if action == "L":
        if column == 0:
            next_row = row
            next_column = column
            return next_row, next_column
        elif np.isnan(initial_state[row][column-1]):
            next_row = row
            next_column = column
            return next_row, next_column
        else:
            next_row = row
            next_column = column - 1
            return next_row, next_column
    if action == "R":
        if column == max_column:
            next_row = row
            next_column = column
            return next_row, next_column
        elif np.isnan(initial_state[row][column+1]):
            next_row = row
            next_column = column
            return next_row, next_column
        else:
            next_row = row
            next_column = column + 1
            return next_row, next_column

#Value Iteration function
def value_iteration(initial_state, reward_matrix, transition_probabilities, discount_rate, size, terminal_states, walls):
    print("-------------------------Value Iteration-------------------------")
    actions = ["U", "D", "L", "R"]
    current_state = deepcopy(initial_state)
    utility = deepcopy(initial_state)
    iterable_utility = deepcopy(initial_state)
    row_length, column_length = current_state.shape
    previous_utility = np.zeros((row_length, column_length))
    final_policy = np.zeros((row_length ,column_length), dtype='U1')
    num_rows, num_columns = size
    action_cell = []
    flag = True
    iteration = 0
    while flag == True:
        #Main Iteration Loop
        iteration = iteration + 1
        utility = deepcopy(iterable_utility)
        for row in range(row_length):
            for column in range(column_length):
                utility_cell = []
                action_tracker = []
                actionoutcome_tracker = []
                #cell_utility variable
                #Cell-wise Operation
                for intended_action in actions:
                    Q = 0
                    action_tracker.append(intended_action)
                    possible_actions, corresponsing_probabilities = action_predictor(row, column, intended_action, transition_probabilities)
                    for q_value in range(len(possible_actions)):
                        next_row, next_column = action_outcome(initial_state, row, column, possible_actions[q_value])
                        q_action = corresponsing_probabilities[q_value] * (reward_matrix[next_row][next_column] + (discount_rate * utility[next_row][next_column]))
                        Q = Q + q_action
                    utility_cell.append(Q)
                    actionoutcome_tracker.append(Q)
                iterable_utility[row][column] = max(utility_cell)
                max_action_index = actionoutcome_tracker.index(max(actionoutcome_tracker))
                final_policy[row][column] = action_tracker[max_action_index]
        for row in range(row_length):
            for column in range(column_length):
                if previous_utility[row][column] != iterable_utility[row][column]:
                    flag = True
                    continue
                else:
                    flag = False
        previous_utility = deepcopy(iterable_utility)
        for value in range(len(terminal_states)):
            state_rows, state_columns = terminal_states[value]
            iterable_utility[abs(state_columns - num_columns)][abs(state_rows - 1)] = 0
        print("\n")
        print("Iteration: ", iteration)
        print(iterable_utility)
    for value in range(len(terminal_states)):
        state_rows, state_columns = terminal_states[value]
        final_policy[abs(state_columns - num_columns)][abs(state_rows - 1)] = 'T'
    for wall in walls:
        wall_row, wall_column = wall
        final_policy[wall_column-1][wall_row-1] = '-'
    print("\n")
    print("Final Policy after Convergence: ")
    print(final_policy)
    return iterable_utility, final_policy

#Policy Iteration function
def policy_iteration(initial_state, reward_matrix, transition_probabilities, discount_rate, size, terminal_states, walls, epsilon):
    actions = ["U", "D", "L", "R"]
    current_state = deepcopy(initial_state)
    utility = deepcopy(initial_state)
    iterable_utility = deepcopy(initial_state)
    row_length, column_length = initial_state.shape
    previous_utility = np.zeros((row_length, column_length))
    num_rows, num_columns = size
    #Initializing Policy Matrix
    policy = np.zeros((row_length ,column_length), dtype='U1')
    random_action = np.random.randint(0, len(actions), (row_length, column_length))
    #Filling Random Actions in Policy to Create Initial Random Policy Matrix
    for row in range(row_length):
        for column in range(column_length):
            policy[row][column] = actions[random_action[row][column]]
    initial_random_policy = deepcopy(policy)
    outer_flag = True
    inner_flag = True
    optimal_utility, optimal_policy = value_iteration(initial_state, reward_matrix, transition_probabilities, discount_rate, size, terminal_states, walls)
    iteration = 0
    while outer_flag is True:
        iteration+=1
        epsilon_array = []
        #Iteration to compare Policy Values
        while inner_flag is True:
            #Value Iteration
            utility = deepcopy(iterable_utility)
            for row in range(row_length):
                for column in range(column_length):
                    intended_action = []
                    intended_action.append(policy[row][column])
                    utility_cell = []
                    for action in intended_action:
                        Q = 0
                        possible_actions, corresponsing_probabilities = action_predictor(row, column, intended_action, transition_probabilities)
                        for q_value in range(len(possible_actions)):
                            next_row, next_column = action_outcome(current_state, row, column, possible_actions[q_value])
                            q_action = corresponsing_probabilities[q_value] * (reward_matrix[next_row][next_column] + (discount_rate * utility[next_row][next_column]))
                            Q = Q + q_action
                        utility_cell.append(Q)
                    iterable_utility[row][column] = max(utility_cell)
            for value in range(len(terminal_states)):
                state_rows, state_columns = terminal_states[value]
                iterable_utility[abs(state_columns - num_columns)][abs(state_rows - 1)] = 0
            for row in range(row_length):
                for column in range(column_length):
                    if previous_utility[row][column] != iterable_utility[row][column]:
                        inner_flag = True
                        continue
                    else:
                        inner_flag = False
            previous_utility = deepcopy(iterable_utility)
            for value in range(len(terminal_states)):
                state_rows, state_columns = terminal_states[value]
                iterable_utility[abs(state_columns - num_columns)][abs(state_rows - 1)] = 0

        for row in range(row_length):
            for column in range(column_length):
                value_difference = optimal_utility[row][column] - iterable_utility[row][column]
                epsilon_array.append(value_difference)

        for row in range(row_length):
            for column in range(column_length):
                if optimal_utility[row][column] > iterable_utility[row][column]:
                    policy[row][column] = optimal_policy[row][column]
                else:
                    continue
        for row in range(row_length):
            for column in range(column_length):
                if policy[row][column] != optimal_policy[row][column]:
                    outer_flag = True
                    continue
                else:
                    outer_flag = False
    for value in range(len(terminal_states)):
        state_rows, state_columns = terminal_states[value]
        policy[abs(state_columns - num_columns)][abs(state_rows - 1)] = 'T'
    for wall in walls:
        wall_row, wall_column = wall
        policy[wall_column-1][wall_row-1] = '-'
    print("------------------------Policy Iteration-------------------------")
    print("\n Initial Random Policy: ")
    print(initial_random_policy)
    print("\n Final Policy: ")
    print(policy)

def main():
    #Obtains extracted data from the Input File
    size, walls, terminal_states, terminal_state_reward, reward, transition_probabilities, discount_rate, epsilon = read_inputfile()
    #Creating Initial State Array
    initial_state = create_initialstate(size, walls)
    #Creating Reward Matrix Array
    reward_matrix = create_rewardmatrix(size, reward, terminal_states, terminal_state_reward)
    #Performing Policy Iteration
    policy_iteration(initial_state, reward_matrix, transition_probabilities, discount_rate, size, terminal_states, walls, epsilon)

if __name__=="__main__":
    main()
