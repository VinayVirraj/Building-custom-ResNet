### YOUR CODE HERE
import torch
import torch.nn as nn
import os, time
import numpy as np
from Network import MyNetwork
from ImageUtils import parse_record

"""This script defines the training, validation and testing process.
"""

class MyModel(nn.Module):

    def __init__(self, configs):
        super(MyModel, self).__init__()
        self.configs = configs
        # self.network = MyNetwork(configs)
        # print(self.network(parse_record()))

    def model_setup(self):
        pass

    def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):
        pass

    def evaluate(self, x, y):
        pass

    def predict_prob(self, x):
        pass


### END CODE HERE