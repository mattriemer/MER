# An implementation of MER Algorithm 1 from https://openreview.net/pdf?id=B1gTShAct7

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

from .common import MLP, ResNet18
import random
from torch.nn.modules.loss import CrossEntropyLoss
from random import shuffle
import sys
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

class Net(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        
        nl, nh = args.n_layers, args.n_hiddens
        
        self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])

        self.bce = CrossEntropyLoss()
        self.n_outputs = n_outputs

        self.opt = optim.SGD(self.parameters(), args.lr)
        self.batchSize = int(args.replay_batch_size)

        self.memories = args.memories
        self.steps = int(args.batches_per_example)
        self.beta = args.beta
        self.gamma = args.gamma

        # allocate buffer
        self.M = []
        self.age = 0
        
        # handle gpus if specified
        self.cuda = args.cuda
        if self.cuda:
            self.net = self.net.cuda()


    def forward(self, x, t):
        output = self.net(x)
        return output

    def getBatch(self,x,y,t):
        xi = Variable(torch.from_numpy(np.array(x))).float().view(1,-1)
        yi = Variable(torch.from_numpy(np.array(y))).long().view(1)
        if self.cuda:
            xi = xi.cuda()
            yi = yi.cuda()
        bxs = [xi]
        bys = [yi]
        
        if len(self.M) > 0:
            order = [i for i in range(0,len(self.M))]
            osize = min(self.batchSize,len(self.M))
            for j in range(0,osize):
                shuffle(order)
                k = order[j]
                x,y,t = self.M[k]
                xi = Variable(torch.from_numpy(np.array(x))).float().view(1,-1)
                yi = Variable(torch.from_numpy(np.array(y))).long().view(1)
                # handle gpus if specified
                if self.cuda:
                    xi = xi.cuda()
                    yi = yi.cuda()
                bxs.append(xi)
                bys.append(yi)
                
                

 
        return bxs,bys
                

    def observe(self, x, t, y):
        ### step through elements of x
        for i in range(0,x.size()[0]):
            self.age += 1
            xi = x[i].data.cpu().numpy()
            yi = y[i].data.cpu().numpy()
            self.net.zero_grad()

            before = deepcopy(self.net.state_dict())
            for step in range(0,self.steps):                
                weights_before = deepcopy(self.net.state_dict())
                # Draw batch from buffer:
                bxs,bys = self.getBatch(xi,yi,t)
                loss = 0.0
                for idx in range(len(bxs)):
                    self.net.zero_grad()
                    bx = bxs[idx]
                    by = bys[idx] 
                    prediction = self.forward(bx,0)
                    loss = self.bce(prediction,by)
                    loss.backward()
                    self.opt.step()
                
                weights_after = self.net.state_dict()
                
                # Within batch Reptile meta-update:
                self.net.load_state_dict({name : weights_before[name] + ((weights_after[name] - weights_before[name]) * self.beta) for name in weights_before})
            
            after = self.net.state_dict()
            
            # Across batch Reptile meta-update:
            self.net.load_state_dict({name : before[name] + ((after[name] - before[name]) * self.gamma) for name in before})

                    

            # Reservoir sampling memory update: 
            
            if len(self.M) < self.memories:
                self.M.append([xi,yi,t])

            else:
                p = random.randint(0,self.age)
                if p < self.memories:
                    self.M[p] = [xi,yi,t]


