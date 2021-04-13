import numpy as np
import torch

import grid_dictionaries as gd
import lrs2_dictionaries as ld


def grid2lrs2labels(grid_embeddings):
    '''
    description:
        Converts grid embeddings to lrs2 embeddings

    inputs:
        grid_embeddings: Pytorch tensor size: (bx6x51) of the grid embeddings

    outputs:
        targetBatch:     Pytorch tensor size: (n) lrs2 targetBatch embedding
        targetLenBatch:  Pytorch tensor size: (n) lrs2 targetLenBatch embedding
    '''
    keys = (grid_embeddings==1).nonzero()[:,-1]

    counter = 0
    curr_len=0
    chars_list = []
    word_list = []
    targetLenBatch=[]
    ## NOTE: I WAS UNSURE ABOUT WHAT TO DO FOR THE targetLenBatch EOS CHAR.
    for key in keys:
        ## NOTE: A counter was used instead of i%6==0 because % is computationally inefficient

        curr_word=gd.grid2chars[key]

        chars_list.append(curr_word.copy())

        if(counter==5):
            chars_list[-1].append("<EOS>")
        else:
            chars_list[-1].append(" ")

        curr_len+=len(curr_word)+1  #A 1 is added for the space of EOS char added to the end
        counter+=1

        if(counter==6):
            targetLenBatch.append(curr_len)
            counter=0
            curr_len=0

    flat_list = [item for sublist in chars_list for item in sublist]

    targetBatch = torch.Tensor([ld.char2idx.get(key) for key in flat_list]).type(torch.LongTensor)
    targetLenBatch = torch.Tensor(targetLenBatch).type(torch.LongTensor)

    return targetBatch, targetLenBatch
















'''#Assuming all are PyTorch Tensors

test_tens = torch.zeros(2,6,51)

#bin blue at a one again
test_tens[[0,0,0,0,0,0],[0,1,2,3,4,5],[0,4,8,12,37,47]]=1

#place green with f six please
test_tens[[1,1,1,1,1,1],[0,1,2,3,4,5],[2,5,11,17,42,49]]=1

(test_tens==1).nonzero()[:,-1]



command2char = {
    0: ['B','I','N'],
    1: ['L','A','Y'],
    2: ['P','L','A','C','E'],
    3: ['S','E','T'],
}

tt=[]
command2char[0]
tt.append(command2char[0].copy())
tt[0].append('2')

char_lists = [command2char.get(key) for key in [0,3,0,2,3,1,0,3,0,2,3,1]]

keys = [0,3,0,2,3,1,0,3,0,2,3,2]

for i, key in enumerate(keys):
    print(i)
    print(key)'''
