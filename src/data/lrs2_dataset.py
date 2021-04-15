"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""

from torch.utils.data import Dataset
from scipy.io import wavfile
import numpy as np
import random

from src.data.lrs2_utils import prepare_pretrain_input
from src.data.lrs2_utils import prepare_main_input



class LRS2Pretrain(Dataset):

    """
    A custom dataset class for the LRS2 pretrain (includes pretain, preval) dataset.
    """

    def __init__(self, dataset, datadir, numWords, charToIx, stepSize, audioParams, videoParams, noiseParams):
        super(LRS2Pretrain, self).__init__()
        with open(datadir + "/" + dataset + ".txt", "r") as f:
            lines = f.readlines()
        self.datalist = [datadir + "/pretrain/" + line.strip() for line in lines]
        self.numWords = numWords
        self.charToIx = charToIx
        self.dataset = dataset
        self.stepSize = stepSize
        self.audioParams = audioParams
        self.videoParams = videoParams
        _, self.noise = wavfile.read(noiseParams["noiseFile"])
        self.noiseProb = noiseParams["noiseProb"]
        self.noiseSNR = noiseParams["noiseSNR"]
        return


    def __getitem__(self, index):
        if self.dataset == "pretrain":
            #index goes from 0 to stepSize-1
            #dividing the dataset into partitions of size equal to stepSize and selecting a random partition
            #fetch the sample at position 'index' in this randomly selected partition
            base = self.stepSize * np.arange(int(len(self.datalist)/self.stepSize)+1)
            ixs = base + index
            ixs = ixs[ixs < len(self.datalist)]
            index = np.random.choice(ixs)

        #passing the sample files and the target file paths to the prepare function to obtain the input tensors
        audioFile = self.datalist[index] + ".wav"
        visualFeaturesFile = self.datalist[index] + ".npy"
        targetFile = self.datalist[index] + ".txt"
        if np.random.choice([True, False], p=[self.noiseProb, 1-self.noiseProb]):
            noise = self.noise
        else:
            noise = None
        inp, trgt, inpLen, trgtLen = prepare_pretrain_input(audioFile, visualFeaturesFile, targetFile, noise, self.numWords,
                                                            self.charToIx, self.noiseSNR, self.audioParams, self.videoParams)
        return inp, trgt, inpLen, trgtLen


    def __len__(self):
        #each iteration covers only a random subset of all the training samples whose size is given by the step size
        #this is done only for the pretrain set, while the whole preval set is considered
        if self.dataset == "pretrain":
            return self.stepSize
        else:
            return len(self.datalist)


class LRS2Main(Dataset):

    """
    A custom dataset class for the LRS2 main (includes train, val, test) dataset
    """

    def __init__(self, dataset, datadir, reqInpLen, charToIx, stepSize, audioParams, videoParams, noiseParams, subset_ratio=0.2):
        super(LRS2Main, self).__init__()
        with open(datadir + "/" + dataset + ".txt", "r") as f:
            lines = f.readlines()
        self.datalist = [datadir + "/main/" + line.strip().split(" ")[0] for line in lines]
        self.reqInpLen = reqInpLen
        self.charToIx = charToIx
        self.dataset = dataset
        self.audioParams = audioParams
        self.videoParams = videoParams
        _, self.noise = wavfile.read(noiseParams["noiseFile"])
        self.noiseSNR = noiseParams["noiseSNR"]
        self.noiseProb = noiseParams["noiseProb"]

        if dataset == "train":
            random.Random(4).shuffle(self.datalist)
            self.datalist = self.datalist[:int(subset_ratio*len(self.datalist))]


    def __getitem__(self, index):

        #passing the sample files and the target file paths to the prepare function to obtain the input tensors
        audioFile = self.datalist[index] + ".wav"
        visualFeaturesFile = self.datalist[index] + ".npy"
        targetFile = self.datalist[index] + ".txt"
        if np.random.choice([True, False], p=[self.noiseProb, 1-self.noiseProb]):
            noise = self.noise
        else:
            noise = None
        inp, trgt, inpLen, trgtLen = prepare_main_input(audioFile, visualFeaturesFile, targetFile, noise, self.reqInpLen, self.charToIx,
                                                        self.noiseSNR, self.audioParams, self.videoParams)

        return inp, trgt, inpLen, trgtLen


    def __len__(self):
        #using step size only for train dataset and not for val and test datasets because
        #the size of val and test datasets is smaller than step size and we generally want to validate and test
        #on the complete dataset
        return len(self.datalist)


if __name__ == "__main__":
    import matplotlib
    import torch
    from src.data.lrs2_config import get_LRS2_Cfg
    from src.data.lrs2_utils import collate_fn
    from torch.utils.data import DataLoader

    args = get_LRS2_Cfg()

    matplotlib.use("Agg")
    np.random.seed(args["SEED"])
    torch.manual_seed(args["SEED"])
    gpuAvailable = torch.cuda.is_available()
    device = torch.device("cuda" if gpuAvailable else "cpu")
    kwargs = {"num_workers": 8, "pin_memory": True} if gpuAvailable else {}
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    #declaring the train and validation datasets and their corresponding dataloaders
    audioParams = {"stftWindow":args["STFT_WINDOW"], "stftWinLen":args["STFT_WIN_LENGTH"], "stftOverlap":args["STFT_OVERLAP"]}
    videoParams = {"videoFPS":args["VIDEO_FPS"]}
    noiseParams = {"noiseFile":args["DATA_DIRECTORY"] + "/noise.wav", "noiseProb":args["NOISE_PROBABILITY"], "noiseSNR":args["NOISE_SNR_DB"]}
    trainData = LRS2Main("train", args["DATA_DIRECTORY"], args["MAIN_REQ_INPUT_LENGTH"], args["CHAR_TO_INDEX"], args["EPOCH_SIZE"],
                         audioParams, videoParams, noiseParams)
    trainLoader = DataLoader(trainData, batch_size=2, collate_fn=collate_fn, shuffle=True, **kwargs)

    data_iter = iter(trainLoader)
    # inp, trgt, inpLen, trgtLen = next(data_iter)
    (inputBatch, targetBatch, inputLenBatch, targetLenBatch) = next(data_iter)
    print(inputBatch[0].shape, inputBatch[1].shape) # ([580, 8, 321]), [145, 8, 512]
    print(targetBatch) # blue 2 g, orange 3 4 -> [, , , , ]
    print(targetLenBatch) # -> [8, 10]
    print("found data")