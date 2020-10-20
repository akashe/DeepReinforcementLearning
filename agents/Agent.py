import torch


class Agent(object):
    def take_action(self,*inputs):
        raise NotImplementedError
