# Created by Chenye Yang on 2020/8/22.
# Copyright © 2020 Chenye Yang. All rights reserved.

import scheduler
import environment
import predictor
import time
from pympler import asizeof

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

if __name__ == '__main__':
    buffer_size = 5 # number of max blocks to store in client buffer
    num_actions = 4 # number of actions / scheduler responses / client requests
    num_block = [3 for i in range(num_actions)]  # each response has 5 blocks
    utility = [[0.6, 0.3, 0.1] for i in range(num_actions)] # utility[response] is the <list> utilities of blocks

    print(color.PURPLE + 'Predefined transformation relationship'.center(80, '#') + color.END)
    print(color.YELLOW + 'INFO: ' + color.END + 'Generating scheduler environment...')
    tic = time.perf_counter()
    schedulerEnv = environment.SchedulerEnv(buffer_size, num_actions, num_block, utility)
    toc = time.perf_counter()
    print(color.GREEN + 'SUCCESS: ' + color.END + color.BLUE + '[{:0.4f} sec] '.format(toc - tic) + color.END + 'Scheduler environment generated! Size: {}'.format(asizeof.asizeof(schedulerEnv)))
    print('Number of states: {}'.format(schedulerEnv.num_states))

    print(color.YELLOW + 'INFO: ' + color.END + 'Generating predictor...')
    tic = time.perf_counter()
    predictorMgm = predictor.Predictor(buffer_size, num_actions)
    toc = time.perf_counter()
    print(color.GREEN + 'SUCCESS: ' + color.END + color.BLUE + '[{:0.4f} sec] '.format(toc - tic) + color.END + 'Predictor generated! Size: {}'.format(asizeof.asizeof(predictorMgm)))

    print(color.YELLOW + 'INFO: ' + color.END + 'Generating scheduler...')
    tic = time.perf_counter()
    schedulerAlg = scheduler.Scheduler()
    toc = time.perf_counter()
    print(color.GREEN + 'SUCCESS: ' + color.END + color.BLUE + '[{:0.4f} sec] '.format(toc - tic) + color.END + 'Scheduler generated! Size: {}'.format(asizeof.asizeof(schedulerAlg)))

    tic = time.perf_counter()
    Q = schedulerAlg.QLearning(predictorMgm.prediction(), schedulerEnv, num_episodes=1000, gamma=1, lr=0.1, e=0.1)
    toc = time.perf_counter()
    print(color.BLUE + '[{:0.4f} sec] '.format(toc - tic) + color.END + 'Action values of Qlearning:')
    print(Q)

    tic = time.perf_counter()
    Q = schedulerAlg.SARSA(predictorMgm.prediction(), schedulerEnv, num_episodes=1000, gamma=1, lr=0.1, e=0.1)
    toc = time.perf_counter()
    print(color.BLUE + '[{:0.4f} sec] '.format(toc - tic) + color.END + 'Action values of SARSA:')
    print(Q)

    # print(color.PURPLE + 'Dynamic transformation relationship'.center(80, '#') + color.END)
    # print(color.YELLOW + 'INFO: ' + color.END + 'Generating dynamic scheduler environment...')
    # tic = time.perf_counter()
    # schedulerEnv_Dynamic = environment.SchedulerEnv_Dynamic(buffer_size, num_actions, num_block, utility)
    # toc = time.perf_counter()
    # print(color.GREEN + 'SUCCESS: ' + color.END + color.BLUE + '[{:0.4f} sec] '.format(toc - tic) + color.END + 'Dynamic Scheduler environment generated! Size: {}'.format(asizeof.asizeof(schedulerEnv_Dynamic)))
    # print('Number of states: {}'.format(schedulerEnv_Dynamic.num_states))
    #
    # print(color.YELLOW + 'INFO: ' + color.END + 'Generating predictor...')
    # tic = time.perf_counter()
    # predictorMgm = predictor.Predictor(buffer_size, num_actions)
    # toc = time.perf_counter()
    # print(color.GREEN + 'SUCCESS: ' + color.END + color.BLUE + '[{:0.4f} sec] '.format(toc - tic) + color.END + 'Predictor generated! Size: {}'.format(asizeof.asizeof(predictorMgm)))
    #
    # print(color.YELLOW + 'INFO: ' + color.END + 'Generating dynamic scheduler...')
    # tic = time.perf_counter()
    # schedulerAlg_Dynamic = scheduler.Scheduler_Dynamic()
    # toc = time.perf_counter()
    # print(color.GREEN + 'SUCCESS: ' + color.END + color.BLUE + '[{:0.4f} sec] '.format(toc - tic) + color.END + 'Dynamic Scheduler generated! Size: {}'.format(asizeof.asizeof(schedulerAlg_Dynamic)))
    #
    # tic = time.perf_counter()
    # Q = schedulerAlg_Dynamic.QLearning(predictorMgm.prediction(), schedulerEnv_Dynamic, num_episodes=1000, gamma=1, lr=0.1, e=0.1)
    # toc = time.perf_counter()
    # print(color.BLUE + '[{:0.4f} sec] '.format(toc - tic) + color.END + 'Action values of Qlearning:')
    # print(Q)
    #
    # tic = time.perf_counter()
    # Q = schedulerAlg_Dynamic.SARSA(predictorMgm.prediction(), schedulerEnv_Dynamic, num_episodes=1000, gamma=1, lr=0.1, e=0.1)
    # toc = time.perf_counter()
    # print(color.BLUE + '[{:0.4f} sec] '.format(toc - tic) + color.END + 'Action values of SARSA:')
    # print(Q)
