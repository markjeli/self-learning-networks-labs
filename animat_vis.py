# animat visualisation
import os
import time
import numpy as np


delay = 0.2   #  delay between frames in sec.



def clear_console():
    os.system('cls')   # MSWindows
    #os.system('clear') # Linux 

def load_map(file_name):
    file_ptr = open(file_name, 'r').read()
    lines = file_ptr.split('\n')
    number_of_lines = lines.__len__() - 1
    row_values = lines[0].split()
    number_of_values = row_values.__len__()

    number_of_rows = 0
    for i in range(number_of_lines):
        row_values = lines[i].split()
        number_of_values = row_values.__len__()
        if (number_of_values > 0):
            number_of_rows += 1
            num_of_columns = number_of_values

    map_of_rew = np.zeros([number_of_rows, num_of_columns], dtype=int)
    #print("examples shape = " + str(map_of_rew.shape))
    
    index = 0
    for i in range(number_of_lines):
        row_values = lines[i].split()
        number_of_values = row_values.__len__()
        if (number_of_values > 0):
            for j in range(number_of_values):
                map_of_rew[index][j] = int(row_values[j])
            index = index + 1

    return map_of_rew

def load_path(file_name):
    file_ptr = open(file_name, 'r').read()
    lines = file_ptr.split('\n')
    number_of_lines = lines.__len__() - 1
    path = []
    for i in range(number_of_lines):
        line = lines[i]
        line2 = ""
        for j in range(len(line)):
            if (line[j] != '[')&(line[j] != ']'):
                line2 += line[j]
        val_str = line2.split(',')
        p = []
        for j in range(len(val_str)-1):
            p.append(int(val_str[j]))
        p.append(float(val_str[-1]))

        path.append(p)
    return path


def visualize():
    actions = [">", "v","<","^"]
    objects = [".","X","p","R","P","r","6","7"]
    for episode in range(5):
        file_name = "map_episode_" + str(episode) + ".txt"
        map = load_map(file_name)
        num_of_rows, num_of_columns = np.shape(map)
        path = load_path("path_episode_" + str(episode) + ".txt")
        clear_console()
        print("\n\n\n\episode " + str(episode))
        time.sleep(2.0)
        for i in range(len(path)): 
            clear_console()
            print("X-wall, p - invisible punish., P - visible pubish, r - invis. reward, R - visible reward\n")
            print("\nstep = " + str(i) + " action = " + str(actions[path[i][2]]) + " reward = " + str([path[i][5]]) )
            print("new state:")
            row = path[i][3]
            column = path[i][4]
            
            for r in range(num_of_rows):
                line = ""
                if (r == row)&(column == 0):
                    znak = "|"
                else:
                    znak = " "
                line += znak
                for c in range(num_of_columns):
                    if (r == row)&(c == column):
                        znak = "|"
                    elif (r == row)&(c+1 == column):
                        znak = "|"
                    else:
                        znak = " "
                    line += objects[map[r,c]] + znak
                print(line)
            time.sleep(delay)

def visualize_fixed_obs():
    actions = [">", "v","<","^"]
    objects = [".","X","p","R","P","r","6","7"]
    d_r = 5 // 2
    d_c = 5 // 2
    for episode in range(5):
        file_name = "map_episode_" + str(episode) + ".txt"
        map = load_map(file_name)
        num_of_rows, num_of_columns = np.shape(map)
        row_obs = num_of_rows //2
        column_obs = num_of_columns //2
        path = load_path("path_episode_" + str(episode) + ".txt")
        clear_console()
        print("\n\n\nepisode " + str(episode))
        time.sleep(2.0)
        for i in range(len(path)): 
            clear_console()
            print("X-wall, p - invisible punish., P - visible pubish, r - invis. reward, R - visible reward\n")
            print("\nstep = " + str(i) + " action = " + str(actions[path[i][2]]) + " reward = " + str([path[i][5]]) )
            print("new state:")
            row = path[i][3]
            column = path[i][4]
            
            map_fixed = np.zeros([num_of_rows, num_of_columns], dtype = int)
            dr = row - row_obs
            dc = column - column_obs
            for r in range(num_of_rows):
                for c in range(num_of_columns):
                    r_obs = r + dr
                    if r_obs < 0:
                        r_obs += num_of_rows
                    elif r_obs >= num_of_rows:
                        r_obs -= num_of_rows
                    c_obs = c + dc
                    if c_obs < 0:
                        c_obs += num_of_columns
                    elif c_obs >= num_of_columns:
                        c_obs -= num_of_columns
                    map_fixed[r,c] = map[r_obs,c_obs]

            for r in range(num_of_rows):
                line = ""
                if (r == row_obs)&(column_obs == 0):
                    znak = "|"
                else:
                    znak = " "
                line += znak
                for c in range(num_of_columns):
                    if (r == row_obs)&(c == column_obs):
                        znak = "|"
                    elif (r == row_obs)&(c+1 == column_obs):
                        znak = "|"
                    else:
                        znak = " "
                    line += objects[map_fixed[r,c]] + znak
                print(line)
            time.sleep(delay)

visualize()
#visualize_fixed_obs()


