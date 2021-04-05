import numpy as np
import torch

from os import listdir
from os.path import isfile, join

## NOTE: Depending on how training is done, the below might need to be tweaked
##       in order to shuffle the training among all speakers. This may also
##       renaming all files to ensure that the names do not overlap (e.g., adding
##       s<x>_ in front of each .npy file)

DATA_GROUP = "s1"

all_input_files = [f for f in listdir('./GRID_DATA/'+DATA_GROUP+'/inputs') if isfile(join('./GRID_DATA/'+DATA_GROUP+'/inputs', f))]
all_label_files = [f for f in listdir('./GRID_DATA/'+DATA_GROUP+'/labels') if isfile(join('./GRID_DATA/'+DATA_GROUP+'/labels', f))]

input_files_list = []
label_files_list = []

for i in range(len(all_input_files)):
    if '.npy' in all_input_files[i]:
        input_files_list.append(all_input_files[i])

for i in range(len(all_label_files)):
    if '.npy' in all_label_files[i]:
        label_files_list.append(all_label_files[i])




class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, input_file_ids, label_file_ids, root_directory):
        'Initialization'
        self.input_file_list = input_file_ids
        self.label_file_list = label_file_ids
        self.root_directory = root_directory


  def __len__(self):
        'Denotes the total number of samples'
        return len(self.label_file_list)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Returns the input as a 75x40x40 array (note that this may need to be
        # changed based on the implementation of the learning algorithm)
        input_ID = self.input_file_list[index]
        label_ID = self.label_file_list[index]

        input_arr = np.load(self.root_directory+'inputs/'+input_ID)
        label_arr = np.load(self.root_directory+'labels/'+label_ID)

        return input_arr, label_arr

params = {'batch_size': 50,
          'shuffle': False}

train_set = Dataset(input_files_list,label_files_list,'./GRID_DATA/s1/')
train_dataloader = torch.utils.data.DataLoader(train_set, **params)

with torch.no_grad():
    for i, batch in enumerate(train_dataloader):
        (curr_input_vecs, curr_label_vecs) = (batch[0].cuda(),batch[1].cuda())

        import pdb; pdb.set_trace()
