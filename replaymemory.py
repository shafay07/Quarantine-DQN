import random
from collections import namedtuple


experience = namedtuple(
    'Experience', ('state', 'action', 'reward', 'nextstate'))


class replaymemory:
    def __init__(self, capacity):
        # capacity defines how much experience to store
        self.capacity = capacity
        self.memory = []  # memory structure to hold experience
        self.push_counter = 0  # counter to see how many experiences we have pushed
        self.batchsize = 50

    # add an experience to memory
    def push(self, experience):
        # check if there is space to add experience
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            # if memory is full start adding to begining to memory
            self.memory[self.push_counter % self.capacity] = experience
        self.push_counter += 1

    # sample the experience
    def sample(self, batchsize):
        return random.sample(self.memory, batchsize)

    # check if we can have sample
    def can_sample(self, batchsize):
        return len(self.memory) >= batchsize
