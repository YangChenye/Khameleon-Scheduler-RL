import random

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

class Predictor():
    def __init__(self, buffer_size, num_requests):
        self.buffer_size = buffer_size
        self.num_actions = num_requests

    def prediction(self):
        # output: the predicted probability distribution of possible requests
        randlist = [random.random() for i in range(self.num_actions)]
        s = sum(randlist)
        pred = [i/s for i in randlist]
        print(color.GREEN + 'SUCCESS: ' + color.END + 'Prediction distribution generated')
        return pred

if __name__ == '__main__':
    # testing
    client = Predictor(5, 3)
    print(client.prediction())