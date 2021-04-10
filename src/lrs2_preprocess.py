"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""

import torch
import numpy as np
import os
import cv2 as cv
from scipy.io import wavfile
from tqdm import tqdm

from data.lrs2_config import get_LRS2_Cfg
from models.deep_avsr.visual_frontend import VisualFrontend

def preprocess_sample(file, params):
    """
    Function to preprocess each data sample.
    available at deep-acsr/src/utils/preprocessing.py
    """

    videoFile = file + ".mp4"
    audioFile = file + ".wav"
    roiFile = file + ".png"
    visualFeaturesFile = file + ".npy"

    roiSize = params["roiSize"]
    normMean = params["normMean"]
    normStd = params["normStd"]
    vf = params["vf"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    #Extract the audio from the video file using the FFmpeg utility and save it to a wav file.
    v2aCommand = "ffmpeg -y -v quiet -i " + videoFile + " -ac 1 -ar 16000 -vn " + audioFile
    os.system(v2aCommand)


    #for each frame, resize to 224x224 and crop the central 112x112 region
    captureObj = cv.VideoCapture(videoFile)
    roiSequence = list()
    while (captureObj.isOpened()):
        ret, frame = captureObj.read()
        if ret == True:
            grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            grayed = grayed/255
            grayed = cv.resize(grayed, (224,224))
            roi = grayed[int(112-(roiSize/2)):int(112+(roiSize/2)), int(112-(roiSize/2)):int(112+(roiSize/2))]
            roiSequence.append(roi)
        else:
            break
    captureObj.release()
    cv.imwrite(roiFile, np.floor(255*np.concatenate(roiSequence, axis=1)).astype(np.int))


    #normalise the frames and extract features for each frame using the visual frontend
    #save the visual features to a .npy file
    inp = np.stack(roiSequence, axis=0)
    inp = np.expand_dims(inp, axis=[1,2])
    inp = (inp - normMean)/normStd
    inputBatch = torch.from_numpy(inp)
    inputBatch = (inputBatch.float()).to(device)
    vf.eval()
    with torch.no_grad():
        outputBatch = vf(inputBatch)
    out = torch.squeeze(outputBatch, dim=1)
    out = out.cpu().numpy()
    np.save(visualFeaturesFile, out)
    return



def main():

    args_lrs2 = get_LRS2_Cfg()
    args_lrs2["DATA_DIRECTORY"] = '../Datasets/LRS2'
    args_lrs2["TRAINED_FRONTEND_FILE"] = 'models/pre-trained_models/deep_avsr_visual_frontend.pt' #absolute path to the trained visual frontend file

    np.random.seed(args_lrs2["SEED"])
    torch.manual_seed(args_lrs2["SEED"])
    gpuAvailable = torch.cuda.is_available()
    device = torch.device("cuda" if gpuAvailable else "cpu")



    #declaring the visual frontend module
    vf = VisualFrontend()
    vf.load_state_dict(torch.load(args_lrs2["TRAINED_FRONTEND_FILE"], map_location=device))
    vf.to(device)


    #walking through the data directory and obtaining a list of all files in the dataset
    filesList = list()
    for root, dirs, files in os.walk(args_lrs2["DATA_DIRECTORY"]):
        for file in files:
            if file.endswith(".mp4"):
                filesList.append(os.path.join(root, file[:-4]))


    #Preprocessing each sample
    print("\nNumber of data samples to be processed = %d" %(len(filesList)))
    print("\n\nStarting preprocessing ....\n")

    params = {"roiSize":args_lrs2["ROI_SIZE"], "normMean":args_lrs2["NORMALIZATION_MEAN"], "normStd":args_lrs2["NORMALIZATION_STD"], "vf":vf}
    for file in tqdm(filesList, leave=True, desc="Preprocess", ncols=75):
        preprocess_sample(file, params)

    print("\nPreprocessing Done.")



    #Generating a 1 hour noise file
    #Fetching audio samples from 20 random files in the dataset and adding them up to generate noise
    #The length of these clips is the shortest audio sample among the 20 samples
    print("\n\nGenerating the noise file ....")

    noise = np.empty((0))
    while len(noise) < 16000*3600:
        noisePart = np.zeros(16000*60)
        indices = np.random.randint(0, len(filesList), 20)
        for ix in indices:
            sampFreq, audio = wavfile.read(filesList[ix] + ".wav")
            audio = audio/np.max(np.abs(audio))
            pos = np.random.randint(0, abs(len(audio)-len(noisePart))+1)
            if len(audio) > len(noisePart):
                noisePart = noisePart + audio[pos:pos+len(noisePart)]
            else:
                noisePart = noisePart[pos:pos+len(audio)] + audio
        noise = np.concatenate([noise, noisePart], axis=0)
    noise = noise[:16000*3600]
    noise = (noise/20)*32767
    noise = np.floor(noise).astype(np.int16)
    wavfile.write(args_lrs2["DATA_DIRECTORY"] + "/noise.wav", 16000, noise)

    print("\nNoise file generated.")



    #Generating preval.txt for splitting the pretrain set into train and validation sets
    # print("\n\nGenerating the preval.txt file ....")
    #
    # with open(args_lrs2["DATA_DIRECTORY"] + "/pretrain.txt", "r") as f:
    #     lines = f.readlines()
    #
    # if os.path.exists(args_lrs2["DATA_DIRECTORY"] + "/preval.txt"):
    #     with open(args_lrs2["DATA_DIRECTORY"] + "/preval.txt", "r") as f:
    #         lines.extend(f.readlines())
    #
    # indices = np.arange(len(lines))
    # np.random.shuffle(indices)
    # valIxs = np.sort(indices[:int(np.ceil(args_lrs2["PRETRAIN_VAL_SPLIT"]*len(indices)))])
    # trainIxs = np.sort(indices[int(np.ceil(args_lrs2["PRETRAIN_VAL_SPLIT"]*len(indices))):])
    #
    # lines = np.sort(np.array(lines))
    # with open(args_lrs2["DATA_DIRECTORY"] + "/pretrain.txt", "w") as f:
    #     f.writelines(list(lines[trainIxs]))
    # with open(args_lrs2["DATA_DIRECTORY"] + "/preval.txt", "w") as f:
    #     f.writelines(list(lines[valIxs]))
    #
    # print("\npreval.txt file generated.\n")

    return



if __name__ == "__main__":
    main()
