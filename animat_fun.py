# animat-like environment - supported functions
import numpy as np

# map generation based on type (0-simplest, 1-simple, ...)
# marks: 
# 0 - empty cell
# 1 - tree or wall
# 2 - invisible punishment (negative reward)
# 3 - visible positive reward 
# 4 - visible punishment (negative reward)
# 5 - invisible positive reward 
# 6 - visible small punishment 
# 7 - visible small positive reward
def generate_map(type):
    # patterns which can be repeated in different places of a map:
    pattern0a = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 4, 1, 3, 1, 4, 1],
        [1, 1, 1, 1, 1, 1, 1],
        ], dtype=int)

    pattern0b = np.array([
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 2, 0, 0, 1],
        [1, 0, 0, 3, 1, 0, 0, 1],
        [1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        ], dtype=int)
    pattern1 = np.array([
        [1, 1, 0],
        [3, 1, 2],
        [0, 1, 1]
        ], dtype=int)
    pattern2 = np.array([
        [1, 2, 1, 0, 1],
        [1, 3, 1, 3, 1],
        [1, 1, 1, 1, 1]
        ], dtype=int)
    pattern3 = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [1, 5, 1, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1]
        ], dtype=int)
    pattern4 = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 5, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1]
        ], dtype=int)
    pattern5 = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 5, 1],
        [1, 1, 1, 1, 1, 1, 1]
        ], dtype=int)
    pattern6 = np.array([
        [0, 0, 0, 1, 1, 0],
        [0, 1, 2, 0, 3, 1],
        [0, 2, 1, 0, 0, 1],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0]
        ], dtype=int)
    pattern7 = np.array([
        [1, 1, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 1, 1]
        ], dtype=int)
    if type == -1:    # static map - always the same - one pattern with reward and punishment
        map = pattern0a
    if type == -2:    # static map - always the same - one pattern with reward and punishment
        map = pattern0b
    elif type == 0:   # one pattern with reward and punishment
        num_of_rows = 7
        num_of_columns = 12
        map = np.zeros((num_of_rows,num_of_columns), dtype=int)
        for i in range(3):
            start_row = np.random.randint(num_of_rows)
            start_column = np.random.randint(num_of_columns)
            num_row_pat, num_col_pat = np.shape(pattern1) 
            for r in range(num_row_pat):
                for c in range(num_col_pat):
                    row = start_row + r
                    if row > num_of_rows-1:
                        row -= num_of_rows
                    col = start_column + c
                    if col > num_of_columns-1:
                        col -= num_of_columns
                    map[row, col] = pattern1[r,c]
    elif type == 1: # one pattern with two similar rewards - one need to pick up punishment 
        num_of_rows = 11
        num_of_columns = 18
        map = np.zeros((num_of_rows,num_of_columns), dtype=int)
        for i in range(3):
            start_row = np.random.randint(num_of_rows)
            start_column = np.random.randint(num_of_columns)
            num_row_pat, num_col_pat = np.shape(pattern2) 
            for r in range(num_row_pat):
                for c in range(num_col_pat):
                    row = start_row + r
                    if row > num_of_rows-1:
                        row -= num_of_rows
                    col = start_column + c
                    if col > num_of_columns-1:
                        col -= num_of_columns
                    map[row, col] = pattern2[r,c]
    elif type == 2: # 3 similar patterns with invisible rewards - agend need to explore all paths
                    # to find reward
        num_of_rows = 14
        num_of_columns = 23
        map = np.zeros((num_of_rows,num_of_columns), dtype=int)
        for i in range(3):
            start_row = np.random.randint(num_of_rows)
            start_column = np.random.randint(num_of_columns)
            patt_no = np.random.randint(3)
            if patt_no == 0:
                pattern = pattern3
            elif patt_no == 1:
                pattern = pattern4
            else:
                pattern = pattern5
            num_row_pat, num_col_pat = np.shape(pattern) 
            for r in range(num_row_pat):
                for c in range(num_col_pat):
                    row = start_row + r
                    if row > num_of_rows-1:
                        row -= num_of_rows
                    col = start_column + c
                    if col > num_of_columns-1:
                        col -= num_of_columns
                    map[row, col] = pattern[r,c]
    elif type == 3: # supermix: all patters mixed randomly in one board
        num_of_rows = 25
        num_of_columns = 40
        map = np.zeros((num_of_rows,num_of_columns), dtype=int)
        num_of_patterns = 5 + np.random.randint(8)
        patterns = [pattern0a, pattern1, pattern2, pattern3, pattern4, pattern5, pattern6, pattern7]
        for i in range(num_of_patterns):
            start_row = np.random.randint(num_of_rows)
            start_column = np.random.randint(num_of_columns)
            patt_no = np.random.randint(len(patterns))
            pattern = patterns[patt_no]
       
            num_row_pat, num_col_pat = np.shape(pattern) 
            for r in range(num_row_pat):
                for c in range(num_col_pat):
                    row = start_row + r
                    if row > num_of_rows-1:
                        row -= num_of_rows
                    col = start_column + c
                    if col > num_of_columns-1:
                        col -= num_of_columns
                    map[row, col] = pattern[r,c]

    return map

# square region (size x size) which agent see from position (row,column) in a given map
# marks: 
# 0 - empty cell
# 1 - tree or wall
# 2 - invisible punishment (negative reward)
# 3 - visible positive reward 
# 4 - visible punishment (negative reward)
# 5 - invisible positive reward 
# 6 - visible small punishment 
# 7 - visible small positive reward
def observable_region(map, size, position, if_cross):
    row, column = position
    region =  np.zeros((size,size), dtype=int)
    num_of_rows, num_of_columns = np.shape(map) 
    for r in range(size):
        for c in range(size):
            r_map = row + r - size//2  
            if r_map < 0:                    # choice of field due to toroidal kind of map 
                r_map += num_of_rows
            elif r_map > num_of_rows - 1:
                r_map -= num_of_rows 
            c_map = column + c - size//2
            if c_map < 0:
                c_map += num_of_columns
            elif c_map > num_of_columns - 1:
                c_map -= num_of_columns
            label = map[r_map, c_map] 
            if (label == 2) | (label == 5):  # removing info about invisible objects
                label = 0
            region[r,c] = label
    # only croos part of the regin is visible:
    if if_cross == True:
        for r in range(size):
            for c in range(size):
                if (r != size//2)&(c != size//2):
                    region[r,c] = -1
    return region

# random start position of an agent - near left side of a map in the middle of vertical dimension,
# not in wall
def start_position(map):
    num_of_rows, num_of_columns = np.shape(map)
    d_row = 1
    d_column = 1
    ok = False
    while ok == False:
        row = 0
        column = np.random.randint(num_of_columns)
        if row < 0:
            row += num_of_rows
        elif row > num_of_rows - 1:
            row -= num_of_rows
        if column < 0:
            column += num_of_columns
        elif column > num_of_columns - 1:
            column -= num_of_columns
        if map[row,column] != 1:
            ok = True
        d_row += 1
        d_column += 1

        if d_row > num_of_rows//2:
            d_row = num_of_rows//2
        if d_column > num_of_columns//2:
            d_column = num_of_columns//2
    return [row,column]

# new agent position and reward after action in true state (row,column): 
# the model of an environment
# possible actions: 0 - right, 1 - down, 2 - left, 3 - up
# the environment can be nondeterministic
def transition_and_reward(map, position, action): 
    row, column = position
    positive_reward = 1.0
    punishment = -1.0
    small_reward = 0.25
    small_punishment = -0.25
    collision_punishment = -0.1

    # transition probabilieties (if == 0 -> deterministic environment)
    prob_side = 0   # probability of moving to the left or right of chosen direction
    prob_back = 0   # probability of moving back of chosen direction 

    prob_forward = 1 - prob_side*2 - prob_back  

    los = np.random.random()    # Random number from uniform distr. from range (0,1)
    right = 0
    down = 1
    left = 2
    up = 3

    if action == right:
        if los < prob_back:
            try_move = left
        elif los < prob_back + prob_side:
            try_move = up
        elif  los < prob_back + 2*prob_side:
            try_move = down
        else:
            try_move = right
    elif action == down:
        if los < prob_back:
            try_move = up
        elif los < prob_back + prob_side:
            try_move = right
        elif  los < prob_back + 2*prob_side:
            try_move = left
        else:
            try_move = down
    elif action == left:
        if los < prob_back:
            try_move = right
        elif los < prob_back + prob_side:
            try_move = up
        elif  los < prob_back + 2*prob_side:
            try_move = down
        else:
            try_move = left
    elif action == up:
        if los < prob_back:
            try_move = down
        elif los < prob_back + prob_side:
            try_move = left
        elif  los < prob_back + 2*prob_side:
            try_move = right
        else:
            try_move = up

    new_row = row
    new_column = column
    if try_move == right:
        new_column += 1
    elif try_move == down:
        new_row += 1
    elif try_move == left:
        new_column -= 1
    elif try_move == up:
        new_row -= 1

    num_of_rows, num_of_columns = np.shape(map)
    if new_row < 0:                   # toroidal world position convertion
        new_row += num_of_rows
    elif new_row > num_of_rows - 1:
        new_row -= num_of_rows

    if new_column < 0:
        new_column += num_of_columns
    elif new_column > num_of_columns - 1:
        new_column -= num_of_columns
    
    reward = 0                         # sum of reward for action in state

    # 0 - empty cell
    # 1 - tree or wall
    # 2 - invisible punishment (negative reward)
    # 3 - visible positive reward 
    # 4 - visible punishment (negative reward)
    # 5 - invisible positive reward 
    # 6 - visible small punishment 
    # 7 - visible small positive reward
    if map[new_row, new_column] == 1:  # collision detected
        reward += collision_punishment
        new_row = row
        new_column = column
    elif (map[new_row, new_column] == 2) | (map[new_row, new_column] == 4):
        reward += punishment
    elif (map[new_row, new_column] == 3) | (map[new_row, new_column] == 5):
        reward += positive_reward
    elif map[new_row, new_column] == 6:
        reward += small_punishment
    elif map[new_row, new_column] == 7:
        reward += small_reward

    return [new_row, new_column], reward

def save_map_and_path(map, path, epi):
    num_of_rows, num_of_columns = np.shape(map)
    filename = "map_episode_" + str(epi) + ".txt"
    f = open(filename,"w")
    for r in range(num_of_rows):
        for c in range(num_of_columns):
            f.write(str(map[r,c]) + " ")
        f.write("\n")
    f.close()
    filename = "path_episode_" + str(epi) + ".txt"
    f = open(filename,"w")
    for i in range(len(path)):
        f.write(str(path[i]) + "\n")
    f.close()

def save_text_animation(map, path, epi):
    num_of_rows, num_of_columns = np.shape(map)
    actions = [">", "v","<","^"]
    objects = [".","X","p","R","P","r","6","7"]
    filename = "text_animation_" + str(epi) + ".txt"
    f = open(filename,"w")
    f.write("X-wall, p - invisible punish., P - visible pubish, r - invis. reward, R - visible reward\n\n")
    for i in range(len(path)): 
        f.write("\nstep = " + str(i) + " action = " + str(actions[path[i][2]]) + " reward = " + str([path[i][5]]) + "\n")
        f.write("new state:\n")
        row = path[i][3]
        column = path[i][4]
        for r in range(num_of_rows):
            if (r == row)&(column == 0):
                znak = "|"
            else:
                znak = " "
            f.write(znak)
            for c in range(num_of_columns):
                if (r == row)&(c == column):
                    znak = "|"
                elif (r == row)&(c+1 == column):
                    znak = "|"
                else:
                    znak = " "
                f.write(objects[map[r,c]] + znak)
            f.write("\n")
    f.close()


