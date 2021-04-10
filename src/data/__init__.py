from src.data.lrs2_dataset import LRS2Main
from src.data.lrs2_config import get_LRS2_Cfg



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

DATA_FACTORY = {
    'LRS2_train': trainData,
    'LRS2_val': valData,
    'LRS2_test': testData,
}


