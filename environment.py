# Created by Chenye Yang on 2020/8/12.
# Copyright © 2020 Chenye Yang. All rights reserved.

from gym.envs.toy_text import discrete
import numpy as np
from scipy.special import comb
from itertools import permutations
from multiprocessing import Pool, cpu_count

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[34m'
   GREEN = '\033[32m'
   YELLOW = '\033[33m'
   RED = '\033[31m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

class SchedulerEnv(discrete.DiscreteEnv):
    """
    The environment of Khameleon
    """
    def __init__(self, buffer_size, num_actions, num_blocks, utility):
        self.buffer_size = buffer_size # number of max blocks to store in client buffer
        self.num_actions = num_actions # number of actions / scheduler responses / client requests
        self.num_blocks = num_blocks # each response has 5 blocks

        # Hypothesis: there's no two same blocks in buffer
        # Hypothesis: the initial buffer is full of blocks, no 00
        self.perms = [p for p in permutations(range(sum(self.num_blocks)), self.buffer_size)]
        # perms are [(0, 1, 2, 3, 4, 5, 6), (0, 1, 2, 3, 4, 5, 7), ...
        # 0-4 means the 5 blocks for first response
        self.num_states = len(self.perms)  # the total number of client buffer state

        # utility[response] is the <list> utilities of blocks
        self.utility = utility

        self.nS = self.num_states
        self.nA = self.num_actions
        # P[state][action] are tuples of tuples tuples with (probability, nextstate, reward, terminal)
        # Hypothesis: no block lost in internet transmit, no delay. Then the prob of next state is 1
        with Pool(processes=cpu_count()) as pool: # initialize the transform matrix with multiprocessing
            self.P = list(pool.map(self.trans, range(self.num_states)))

        # # initialize the transform matrix with single thread
        # self.P = [[[] for action in range(self.num_actions)] for state in range(self.num_states)]
        # for current_s_idx in range(self.num_states):
        #     for action_idx in range(self.num_actions):
        #         prob = 1
        #         next_s_idx = self.next_state_idx(current_s_idx, action_idx)
        #         next_s = self.perms[next_s_idx]
        #         arrival_b_num = next_s[self.buffer_size - 1]
        #         reward = self.reward(arrival_b_num)
        #         terminal = 0
        #         self.P[current_s_idx][action_idx].append((prob, next_s_idx, reward, terminal))

    def block_decode(self, block_num):
        # input: the number in self.perms, each number represent a combination of response and block.
        # output: the response-block pair, indicating which response and which block of the response
        # [0,3] means the 4th block of 1st response (index starts from 0)
        # 3 means the 4th block of 1st response
        # if input 3, output [0,3]
        start_index = [0] # the index number of the first block of each response
        for i in range(self.num_actions - 1):
            start_index.append(start_index[i] + self.num_blocks[i])
        # compare to get the action index
        action = self.num_actions - 1
        for i in range(self.num_actions):
            if block_num < start_index[i]:
                action = i - 1
                break
        # get the block index
        block = block_num - start_index[action]
        # combine
        response_block_pair = [action, block]
        return response_block_pair

    def block_encode(self, response_block_pair):
        # input: the response-block pair, indicating which response and which block of the response
        # output: the number in self.perms, each number represent a combination of response and block.
        # [0,3] means the 4th block of 1st response (index starts from 0)
        # 3 means the 4th block of 1st response
        # if input [0,3], output 3
        start_index = [0] # the index number of the first block of each response
        for i in range(self.num_actions - 1):
            start_index.append(start_index[i] + self.num_blocks[i])
        # split
        action, block = response_block_pair
        # compute the block number
        block_num = start_index[action] + block
        return block_num

    def next_state_idx(self, current_state_index, action):
        # input: the index of current state in self.perms, the index of action
        # output: the index of next state in self.perms
        current_state = self.perms[current_state_index] # (0, 1, 2, 3, 4, 5, 6)
        # create a statistic to count whether the block has appeared in current state
        stat = [[0 for j in range(self.num_blocks[i])] for i in range(self.num_actions)]
        for block_num in current_state:
            a, b = self.block_decode(block_num)
            stat[a][b] = 1 # the action and block has appeared
        # [[1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        stat_sum = list(map(sum, stat))
        # [5, 2, 0, 0, 0, 0]
        if stat_sum[action] == self.num_blocks[action]:
            next_state_index = -1
            # print(color.YELLOW + 'INFO: ' + color.END + 'The buffer has all blocks for the request, no block is sent.')
        else:
            block = stat[action].index(0)
            next_block = self.block_encode([action, block])
            next_state = current_state[1:self.buffer_size] + (next_block,)
            try:
                next_state_index = self.perms.index(next_state)
            except ValueError:
                next_state_index = -2
                print(
                    color.RED + 'ERROR: ' + color.END + 'The next state {} not in the combinations'.format(next_state))
        return next_state_index

    def reward(self, block_num):
        # input: the number in self.perms, each number represent a combination of response and block.
        # output: the utility of this block
        response, block = self.block_decode(block_num)
        r = self.utility[response][block]
        return r

    def trans(self, current_s_idx):
        # input: the index of current state
        # output: [[result of 1st action], [result of 2nd action], [result of 3rd action], [result of 4th action]]
        # [[(prob, next_s_idx, reward, terminal), (prob, next_s_idx, reward, terminal)], [...], [...], [...]]
        p = [[] for action_idx in range(self.num_actions)]
        for action_idx in range(self.num_actions):
            prob = 1
            '''ERROR HERE!!!!! Need to deal with the index=-1, which means nothing changed'''
            next_s_idx = self.next_state_idx(current_s_idx, action_idx)
            next_s = self.perms[next_s_idx]
            arrival_b_num = next_s[self.buffer_size - 1]
            reward = self.reward(arrival_b_num)
            terminal = 0
            p[action_idx].append((prob, next_s_idx, reward, terminal))
        return p

    def buffer_value(self, state_index):
        # input: index of state in self.perms
        # output: the state value of the input state
        buffer = self.perms[state_index]
        b_value = sum(list(map(self.reward, buffer)))
        return b_value

    def list_str(self, state_list):
        # input: state_list: [11,12,13,14,15,21,0]
        # 12 means this buffer position stores the 2nd block of the 1st response
        # 0 means this buffer position is empty
        # output: state_str: '11121314152100'
        state_temp = list(map(str, state_list))
        state_temp = [s.zfill(2) for s in state_temp] # change '0' to '00'
        state_str = ''.join(state_temp)
        return state_str

    def str_list(self, state_str):
        # input: state_str: '11121314152100'
        # '12' means this buffer position stores the 2nd block of the 1st response
        # '00' means this buffer position is empty
        # output: state_list: [11,12,13,14,15,21,0]
        state_list = []
        for i in range(int(state_str.__len__()/2)):
            state_list.append(state_str[0+2*i:2+2*i])
        state_list = list(map(int, state_list))
        return state_list

if __name__ == '__main__':
    # testing
    buffer_size = 5  # number of max blocks to store in client buffer
    num_actions = 4  # number of actions / scheduler responses / client requests
    num_block = [3 for i in range(num_actions)]  # each response has 5 blocks
    utility = [[0.6, 0.3, 0.1] for i in range(num_actions)]  # utility[response] is the <list> utilities of blocks

    schedulerEnv = SchedulerEnv(buffer_size, num_actions, num_block, utility)
    # print(schedulerEnv.list_str((11,12,13,14,15,21,0)))
    # print(schedulerEnv.str_list('11121314152100'))
    print('Num of states: {}'.format(schedulerEnv.num_states))
    # print(schedulerEnv.block_decode(2))
    # print(schedulerEnv.block_encode([2, 2]))

    next_a = 2
    current_s_index = 7
    current_s = schedulerEnv.perms[current_s_index]
    print('The current state: {}'.format(current_s))
    next_s_index = schedulerEnv.next_state_idx(current_s_index, next_a)
    next_s = schedulerEnv.perms[next_s_index]
    print('The next state: {}'.format(next_s))
    arrival_b = schedulerEnv.block_decode(next_s[schedulerEnv.buffer_size - 1])[1]
    print('The new arrival block for request {}: {}'.format(next_a, arrival_b))
    buffer_v = schedulerEnv.buffer_value(current_s_index)
    print('The value of client buffer {} is {}'.format(current_s, buffer_v))