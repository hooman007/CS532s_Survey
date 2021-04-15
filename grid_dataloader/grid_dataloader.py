import numpy as np
import torch

from os import listdir
from os.path import isfile, join
from src.data.lrs2_utils import collate_fn
import torch
from torch.nn.utils.rnn import pad_sequence
from scipy import signal
from scipy.io import wavfile
import cv2 as cv
from scipy.special import softmax
from src.models.deep_avsr.visual_frontend import VisualFrontend
from grid_dataloader.grid2lrs2labels import grid2lrs2labels
from torch.utils.data import DataLoader, random_split

import grid_dataloader.grid_dictionaries as gd
import grid_dataloader.lrs2_dictionaries as ld


## NOTE: Depending on how training is done, the below might need to be tweaked
##       in order to shuffle the training among all speakers. This may also
##       renaming all files to ensure that the names do not overlap (e.g., adding
##       s<x>_ in front of each .npy file)

group = ["s1", "s2"]
all_input_files = []
all_label_files = []
for DATA_GROUP in group:
    all_input_files.extend([join('./grid_dataloader/GRID_DATA/'+DATA_GROUP+'/inputs', f) for f in listdir('./grid_dataloader/GRID_DATA/'+DATA_GROUP+'/inputs') if isfile(join('./grid_dataloader/GRID_DATA/'+DATA_GROUP+'/inputs', f))])
    all_label_files.extend([join('./grid_dataloader/GRID_DATA/'+DATA_GROUP+'/labels', f) for f in listdir('./grid_dataloader/GRID_DATA/'+DATA_GROUP+'/labels') if isfile(join('./grid_dataloader/GRID_DATA/'+DATA_GROUP+'/labels', f))])

input_files_list = []
audio_files_list = []
label_files_list = []

for i in range(len(all_input_files)):
    if '.npy' in all_input_files[i]:
        input_files_list.append(all_input_files[i])
    if '.wav' in all_input_files[i]:
        audio_files_list.append(all_input_files[i])


for i in range(len(all_label_files)):
    if '.npy' in all_label_files[i]:
        label_files_list.append(all_label_files[i])


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, input_file_ids, audio_file_ids, label_file_ids, root_directory):
        'Initialization'
        self.input_file_list = input_file_ids
        self.audio_file_list = audio_file_ids
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



class Grid_vf(torch.utils.data.Dataset):

    """
    A custom dataset class for the grid (includes train, val, test) dataset
    """

    def __init__(self,  input_file_ids, audio_file_ids, label_file_ids):
        super(Grid_vf, self).__init__()
        _, self.noise = wavfile.read('data/LRS2/noise.wav')
        self.noiseSNR = 0
        self.noiseProb = 0.25

        self.input_file_list = input_file_ids
        self.audio_file_list = audio_file_ids
        self.label_file_list = label_file_ids
        return


    def __getitem__(self, index):
        input_ID = self.input_file_list[index]
        label_ID = self.label_file_list[index]
        audio_ID = self.audio_file_list[index]

        label_arr = np.load(label_ID) # batch_size, 6, 51
        label_arr = torch.from_numpy(np.expand_dims(label_arr, 0))
        trgt, trgtLen = grid2lrs2labels(label_arr)
        trgtLen = torch.squeeze(trgtLen)

        # get these from Robert

        # STFT feature extraction
        audioFile = audio_ID
        stftWindow = "hamming"
        stftWinLen = 0.040
        stftOverlap = 0.030
        sampFreq, inputAudio = wavfile.read(audioFile)

        # pad the audio to get atleast 4 STFT vectors
        if len(inputAudio) < sampFreq * (stftWinLen + 3 * (stftWinLen - stftOverlap)):
            padding = int(np.ceil((sampFreq * (stftWinLen + 3 * (stftWinLen - stftOverlap)) - len(inputAudio)) / 2))
            inputAudio = np.pad(inputAudio, padding, "constant")
        inputAudio = inputAudio / np.max(np.abs(inputAudio))

        # adding noise to the audio
        # if self.noise is not None:
        #     pos = np.random.randint(0, len(noise) - len(inputAudio) + 1)
        #     noise = noise[pos:pos + len(inputAudio)]
        #     noise = noise / np.max(np.abs(noise))
        #     gain = 10 ** (noiseSNR / 10)
        #     noise = noise * np.sqrt(np.sum(inputAudio ** 2) / (gain * np.sum(noise ** 2)))
        #     inputAudio = inputAudio + noise

        # normalising the audio to unit power
        inputAudio = inputAudio / np.sqrt(np.sum(inputAudio ** 2) / len(inputAudio))

        # computing STFT and taking only the magnitude of it
        _, _, stftVals = signal.stft(inputAudio, sampFreq, window=stftWindow, nperseg=sampFreq * stftWinLen,
                                     noverlap=sampFreq * stftOverlap,
                                     boundary=None, padded=False)
        audInp = np.abs(stftVals)
        audInp = audInp.T

        # loading the visual features
        vidInp = np.load(input_ID)

        # padding zero vectors to extend the audio and video length to a least possible integer length such that
        # video length = 4 * audio length
        if len(audInp) / 4 >= len(vidInp):
            inpLen = int(np.ceil(len(audInp) / 4))
            leftPadding = int\
                (np.floor((4 * inpLen - len(audInp)) / 2))
            rightPadding = int(np.ceil((4 * inpLen - len(audInp)) / 2))
            audInp = np.pad(audInp, ((leftPadding, rightPadding), (0, 0)), "constant")
            leftPadding = int(np.floor((inpLen - len(vidInp)) / 2))
            rightPadding = int(np.ceil((inpLen - len(vidInp)) / 2))
            vidInp = np.pad(vidInp, ((leftPadding, rightPadding), (0, 0)), "constant")
        else:
            inpLen = len(vidInp)
            leftPadding = int(np.floor((4 * inpLen - len(audInp)) / 2))
            rightPadding = int(np.ceil((4 * inpLen - len(audInp)) / 2))
            audInp = np.pad(audInp, ((leftPadding, rightPadding), (0, 0)), "constant")

        # # checking whether the input length is greater than or equal to the required length
        # # if not, extending the input by padding zero vectors
        # reqInpLen = 10
        # if inpLen < reqInpLen:
        #     leftPadding = int(np.floor((reqInpLen - inpLen) / 2))
        #     rightPadding = int(np.ceil((reqInpLen - inpLen) / 2))
        #     audInp = np.pad(audInp, ((4 * leftPadding, 4 * rightPadding), (0, 0)), "constant")
        #     vidInp = np.pad(vidInp, ((leftPadding, rightPadding), (0, 0)), "constant")

        inpLen = len(vidInp)

        audInp = torch.from_numpy(audInp)
        vidInp = torch.from_numpy(vidInp)
        inp = (audInp, vidInp)
        inpLen = torch.tensor(inpLen)
        # if targetFile is not None:
        #     trgt = torch.from_numpy(trgt)
        #     trgtLen = torch.tensor(trgtLen)
        # else:
        #     trgt, trgtLen = None, None
        #

        return inp, trgt, inpLen, trgtLen


    def __len__(self):
        #using step size only for train dataset and not for val and test datasets because
        #the size of val and test datasets is smaller than step size and we generally want to validate and test
        #on the complete dataset
        return len(self.input_file_list)


#
#
#
# gpuAvailable = torch.cuda.is_available()
# print(len(input_files_list), len(audio_files_list), len(label_files_list))
# train_set = Grid_vf(input_files_list, audio_files_list, label_files_list)
# kwargs = {"num_workers": 1, "pin_memory": True} if gpuAvailable else {}
# trainLoader = DataLoader(train_set, batch_size=4, shuffle=False, collate_fn=collate_fn, **kwargs)
# # train_set = Dataset(input_files_list,label_files_list,'./GRID_DATA/s1/')
# # train_dataloader = torch.utils.data.DataLoader(train_set, **params)
#
# data_iter = iter(trainLoader)
#     # inp, trgt, inpLen, trgtLen = next(data_iter)
# (inputBatch, targetBatch, inputLenBatch, targetLenBatch) = next(data_iter)
# print(inputBatch[0].shape, inputBatch[1].shape) # ([580, 8, 321]), [145, 8, 512]
# print(targetBatch) # blue 2 g, orange 3 4 -> [, , , , ]
# print(targetLenBatch.shape) # -> [8, 10]
# print("found data")