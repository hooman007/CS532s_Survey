import numpy as np
import torch

from os import listdir
from os.path import isfile, join

import torch
from torch.nn.utils.rnn import pad_sequence
from scipy import signal
from scipy.io import wavfile
import cv2 as cv
from scipy.special import softmax
from models.deep_avsr.visual_frontend import VisualFrontend

## NOTE: Depending on how training is done, the below might need to be tweaked
##       in order to shuffle the training among all speakers. This may also
##       renaming all files to ensure that the names do not overlap (e.g., adding
##       s<x>_ in front of each .npy file)

DATA_GROUP = "s1"

all_input_files = [f for f in listdir('./GRID_DATA/'+DATA_GROUP+'/inputs') if isfile(join('./GRID_DATA/'+DATA_GROUP+'/inputs', f))]
all_label_files = [f for f in listdir('./GRID_DATA/'+DATA_GROUP+'/labels') if isfile(join('./GRID_DATA/'+DATA_GROUP+'/labels', f))]

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



class Grid_vf(Dataset):

    """
    A custom dataset class for the grid (includes train, val, test) dataset
    """

    def __init__(self, dataset, datadir, reqInpLen, charToIx, stepSize, audioParams, videoParams, noiseParams):
        super(Grid_vf, self).__init__()
        # with open(datadir + "/" + dataset + ".txt", "r") as f:
        #     lines = f.readlines()
        # self.datalist = [datadir + "/main/" + line.strip().split(" ")[0] for line in lines]
        # self.reqInpLen = reqInpLen
        # self.charToIx = charToIx
        # self.dataset = dataset
        # self.stepSize = stepSize
        # self.audioParams = audioParams
        # self.videoParams = videoParams
        _, self.noise = wavfile.read('data/LRS2/noise.wav')
        self.noiseSNR = 0
        self.noiseProb = 0.25

        self.input_file_list = input_file_ids
        self.audio_file_list = audio_file_ids
        self.label_file_list = label_file_ids

        self.root_directory = root_directory
        return


    def __getitem__(self, index):
        #using the same procedure as in pretrain dataset class only for the train dataset
        # if self.dataset == "train":
        #     base = self.stepSize * np.arange(int(len(self.datalist)/self.stepSize)+1)
        #     ixs = base + index
        #     ixs = ixs[ixs < len(self.datalist)]
        #     index = np.random.choice(ixs)

        #passing the sample files and the target file paths to the prepare function to obtain the input tensors
        input_ID = self.input_file_list[index]
        label_ID = self.label_file_list[index]
        audio_ID = self.audio_file_list[index]

        # label_arr = np.load(self.root_directory + 'labels/' + label_ID) # batch_size, 6, 51
        audioFile = self.root_directory + 'inputs/' + audio_ID

        trgt = []
        trgtLen = []
        # get these from Robert

        # STFT feature extraction
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
        if self.noise is not None:
            pos = np.random.randint(0, len(noise) - len(inputAudio) + 1)
            noise = noise[pos:pos + len(inputAudio)]
            noise = noise / np.max(np.abs(noise))
            gain = 10 ** (noiseSNR / 10)
            noise = noise * np.sqrt(np.sum(inputAudio ** 2) / (gain * np.sum(noise ** 2)))
            inputAudio = inputAudio + noise

        # normalising the audio to unit power
        inputAudio = inputAudio / np.sqrt(np.sum(inputAudio ** 2) / len(inputAudio))

        # computing STFT and taking only the magnitude of it
        _, _, stftVals = signal.stft(inputAudio, sampFreq, window=stftWindow, nperseg=sampFreq * stftWinLen,
                                     noverlap=sampFreq * stftOverlap,
                                     boundary=None, padded=False)
        audInp = np.abs(stftVals)
        audInp = audInp.T

        # loading the visual features
        vidInp = np.load(self.root_directory + 'inputs/' + input_ID)

        # padding zero vectors to extend the audio and video length to a least possible integer length such that
        # video length = 4 * audio length
        if len(audInp) / 4 >= len(vidInp):
            inpLen = int(np.ceil(len(audInp) / 4))
            leftPadding = int(np.floor((4 * inpLen - len(audInp)) / 2))
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

        # checking whether the input length is greater than or equal to the required length
        # if not, extending the input by padding zero vectors
        if inpLen < reqInpLen:
            leftPadding = int(np.floor((reqInpLen - inpLen) / 2))
            rightPadding = int(np.ceil((reqInpLen - inpLen) / 2))
            audInp = np.pad(audInp, ((4 * leftPadding, 4 * rightPadding), (0, 0)), "constant")
            vidInp = np.pad(vidInp, ((leftPadding, rightPadding), (0, 0)), "constant")

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
        if self.dataset == "train":
            return self.stepSize
        else:
            return len(self.datalist)



params = {'batch_size': 50,
          'shuffle': False}

train_set = Dataset(input_files_list,label_files_list,'./GRID_DATA/s1/')
train_dataloader = torch.utils.data.DataLoader(train_set, **params)

with torch.no_grad():
    for i, batch in enumerate(train_dataloader):
        (curr_input_vecs, curr_label_vecs) = (batch[0].cuda(),batch[1].cuda())

        import pdb; pdb.set_trace()
