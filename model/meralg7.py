# An implementation of Meta-Experience Replay (MER) Algorithm 7 from https://openreview.net/pdf?id=B1gTShAct7

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
from random import shuffle

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
        self.s = float(args.s)
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
        mxi = Variable(torch.from_numpy(np.array(x))).float().view(1,-1)
        myi = Variable(torch.from_numpy(np.array(y))).long().view(1)
        if self.cuda:
            mxi = mxi.cuda()
            myi = myi.cuda()
        bxs = [mxi]
        bys = [myi]
        
        if len(self.M) > 0:
            order = [i for i in range(0,len(self.M))]
            osize = min(self.batchSize,len(self.M))
            for j in range(0,osize):
                shuffle(order)
                k = order[j]
                x,y,t = self.M[k]
                xi = Variable(torch.from_numpy(np.array(x))).float().view(1,-1)
                yi = Variable(torch.from_numpy(np.array(y))).long().view(1)
                
                if self.cuda:
                    xi = xi.cuda()
                    yi = yi.cuda()
                    
                bxs.append(xi)
                bys.append(yi)

        bx2 = []
        by2 = []
        indexes = [ind for ind in range(len(bxs))]
        shuffle(indexes)
        for index in indexes:
            if index == 0:
                myindex = len(bx2)
            bx2.append(bxs[index])
            by2.append(bys[index])
 
        return bx2,by2,myindex
                

    def observe(self, x, t, y):
        ### step through elements of x
        for i in range(0,x.size()[0]):
            self.age += 1
            xi = x[i].data.cpu().numpy()
            yi = y[i].data.cpu().numpy()
            if self.age > 1:
                self.net.zero_grad()


   
                weights_before = deepcopy(self.net.state_dict())
    
                # Draw batch from buffer:
                bxs,bys,myind = self.getBatch(xi,yi,t)
                
                # SGD on individual samples from batch:
                loss = 0.0
                for idx in range(len(bxs)):
                    self.net.zero_grad()
                    bx = bxs[idx]
                    by = bys[idx] 
                    prediction = self.forward(bx,0)
                    if myind == idx:
                        # High learning rate SGD on current example:
                        loss = self.s*self.bce(prediction,by)
                    else:
                        loss = self.bce(prediction,by)
                    loss.backward()
                    self.opt.step()
                
                weights_after = self.net.state_dict()
                # Reptile meta-update:
                self.net.load_state_dict({name : weights_before[name] + ((weights_after[name] - weights_before[name]) * self.gamma) for name in weights_before})
            
                    

            sys.stdout.flush()
            
            # Reservoir sampling memory update:
            if len(self.M) < self.memories:
                self.M.append([xi,yi,t])

            else:
                p = random.randint(0,self.age)
                if p < self.memories:
                    self.M[p] = [xi,yi,t]


