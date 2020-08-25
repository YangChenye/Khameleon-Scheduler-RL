import scheduler
import environment
import predictor

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

    print(color.YELLOW + 'INFO: ' + color.END + 'Generating scheduler environment...')
    schedulerEnv = environment.SchedulerEnv(buffer_size, num_actions, num_block, utility)
    print(color.GREEN + 'SUCCESS: ' + color.END + 'Scheduler environment generated!')
    print('Number of states: {}'.format(schedulerEnv.num_states))

    print(color.YELLOW + 'INFO: ' + color.END + 'Generating predictor...')
    predictorMgm = predictor.Predictor(buffer_size, num_actions)
    print(color.GREEN + 'SUCCESS: ' + color.END + 'Predictor generated!')

    print(color.YELLOW + 'INFO: ' + color.END + 'Generating scheduler...')
    schedulerAlg = scheduler.Scheduler()
    print(color.GREEN + 'SUCCESS: ' + color.END + 'Scheduler generated!')

    # Q = schedulerAlg.QLearning(env=schedulerEnv, num_episodes=1000, gamma=1, lr=0.1, e=0.1)
    # print('Action values:')
    # print(Q)

