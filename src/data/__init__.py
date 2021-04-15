from src.data.lrs2_dataset import LRS2Main
from src.data.lrs2_config import get_LRS2_Cfg
from grid_dataloader.grid_dataloader import Grid_vf

from os import listdir
from os.path import isfile, join


args = get_LRS2_Cfg()
audioParams = {"stftWindow": args["STFT_WINDOW"], "stftWinLen": args["STFT_WIN_LENGTH"],
               "stftOverlap": args["STFT_OVERLAP"]}
videoParams = {"videoFPS": args["VIDEO_FPS"]}
noiseParams = {"noiseFile": args["DATA_DIRECTORY"] + "/noise.wav", "noiseProb": args["NOISE_PROBABILITY"],
               "noiseSNR": args["NOISE_SNR_DB"]}
trainData = LRS2Main("train", args["DATA_DIRECTORY"], args["MAIN_REQ_INPUT_LENGTH"], args["CHAR_TO_INDEX"],
                     args["EPOCH_SIZE"], audioParams, videoParams, noiseParams)
valData = LRS2Main("val", args["DATA_DIRECTORY"], args["MAIN_REQ_INPUT_LENGTH"], args["CHAR_TO_INDEX"],
                     args["EPOCH_SIZE"], audioParams, videoParams, noiseParams)
testData = LRS2Main("val", args["DATA_DIRECTORY"], args["MAIN_REQ_INPUT_LENGTH"], args["CHAR_TO_INDEX"],
                     args["EPOCH_SIZE"], audioParams, videoParams, noiseParams)



Train_group = ["s{}".format(i) for i in range(1, 8)]
Val_group = ["s8", "s9"]

Train_input_files = []
Train_label_files = []
for speaker in Train_group:
    Train_input_files.extend([join('./grid_dataloader/GRID_DATA/'+speaker+'/inputs', f) for f in listdir('./grid_dataloader/GRID_DATA/'+speaker+'/inputs') if isfile(join('./grid_dataloader/GRID_DATA/'+speaker+'/inputs', f))])
    Train_label_files.extend([join('./grid_dataloader/GRID_DATA/'+speaker+'/labels', f) for f in listdir('./grid_dataloader/GRID_DATA/'+speaker+'/labels') if isfile(join('./grid_dataloader/GRID_DATA/'+speaker+'/labels', f))])

Val_input_files = []
Val_label_files = []
for speaker in Val_group:
    Val_input_files.extend([join('./grid_dataloader/GRID_DATA/'+speaker+'/inputs', f) for f in listdir('./grid_dataloader/GRID_DATA/'+speaker+'/inputs') if isfile(join('./grid_dataloader/GRID_DATA/'+speaker+'/inputs', f))])
    Val_label_files.extend([join('./grid_dataloader/GRID_DATA/'+speaker+'/labels', f) for f in listdir('./grid_dataloader/GRID_DATA/'+speaker+'/labels') if isfile(join('./grid_dataloader/GRID_DATA/'+speaker+'/labels', f))])

train_input_files_list = []
train_audio_files_list = []
train_label_files_list = []

val_input_files_list = []
val_audio_files_list = []
val_label_files_list = []

for i in range(len(Train_input_files)):
    if '.npy' in Train_input_files[i]:
        train_input_files_list.append(Train_input_files[i])
    if '.wav' in Train_input_files[i]:
        train_audio_files_list.append(Train_input_files[i])
for i in range(len(Val_input_files)):
    if '.npy' in Val_input_files[i]:
        val_input_files_list.append(Val_input_files[i])
    if '.wav' in Val_input_files[i]:
        val_audio_files_list.append(Val_input_files[i])

for i in range(len(Train_label_files)):
    if '.npy' in Train_label_files[i]:
        train_label_files_list.append(Train_label_files[i])
for i in range(len(Val_label_files)):
    if '.npy' in Val_label_files[i]:
        val_label_files_list.append(Val_label_files[i])


grid_trainData = Grid_vf(train_input_files_list, train_audio_files_list, train_label_files_list)
grid_valData = Grid_vf(val_input_files_list, val_audio_files_list, val_label_files_list)


DATA_FACTORY = {
    'LRS2_train': trainData,
    'LRS2_val': valData,
    'LRS2_test': testData,
    'grid_train': grid_trainData,
    'grid_val': grid_valData,

}


