"""
Contains the configurations regarding the LRS2 dataset.
Based on the file config.py part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""

def get_LRS2_Cfg():

	args_lrs2 = dict()

	###########################################
	############ project structure ############
	###########################################
	#absolute path to the data directory which contains: 
	# folders of: "main/", "pretrain/"
	# files of "pretrain.txt", "train.txt", "test.txt", "val.txt"
	args_lrs2["DATA_DIRECTORY"] = 'data/LRS2'
	args_lrs2["TRAINED_FRONTEND_FILE"] = 'models/pre-trained_models/deep_avsr_visual_frontend.pt' #absolute path to the trained visual frontend file

	###########################################
	################ data #####################
	###########################################
	args_lrs2["PRETRAIN_VAL_SPLIT"] = 0.01   #validation set size fraction during pretraining
	args_lrs2["PRETRAIN_NUM_WORDS"] = 1  #number of words limit in current curriculum learning iteration
	args_lrs2["MAIN_REQ_INPUT_LENGTH"] = 145 #minimum input length while training
	args_lrs2["CHAR_TO_INDEX"] = {" ":1, "'":22, "1":30, "0":29, "3":37, "2":32, "5":34, "4":38, "7":36, "6":35, "9":31, "8":33,
		                          "A":5, "C":17, "B":20, "E":2, "D":12, "G":16, "F":19, "I":6, "H":9, "K":24, "J":25, "M":18,
		                          "L":11, "O":4, "N":7, "Q":27, "P":21, "S":8, "R":10, "U":13, "T":3, "W":15, "V":23, "Y":14,
		                          "X":26, "Z":28, "<EOS>":39}    #character to index mapping
	args_lrs2["INDEX_TO_CHAR"] = {1:" ", 22:"'", 30:"1", 29:"0", 37:"3", 32:"2", 34:"5", 38:"4", 36:"7", 35:"6", 31:"9", 33:"8",
		                          5:"A", 17:"C", 20:"B", 2:"E", 12:"D", 16:"G", 19:"F", 6:"I", 9:"H", 24:"K", 25:"J", 18:"M",
		                          11:"L", 4:"O", 7:"N", 27:"Q", 21:"P", 8:"S", 10:"R", 13:"U", 3:"T", 15:"W", 23:"V", 14:"Y",
		                          26:"X", 28:"Z", 39:"<EOS>"}    #index to character reverse mapping

	###########################################
	######### Audio preprocessing #############
	###########################################
	args_lrs2["NOISE_PROBABILITY"] = 0.25    #noise addition probability while training
	args_lrs2["NOISE_SNR_DB"] = 0    #noise level in dB SNR
	args_lrs2["STFT_WINDOW"] = "hamming" #window to use while computing STFT
	args_lrs2["STFT_WIN_LENGTH"] = 0.040 #window size in secs for computing STFT
	args_lrs2["STFT_OVERLAP"] = 0.030    #consecutive window overlap in secs while computing STFT

	###########################################
	######### Video preprocessing #############
	###########################################
	args_lrs2["VIDEO_FPS"] = 25  #frame rate of the video clips
	args_lrs2["ROI_SIZE"] = 112  #height and width of input greyscale lip region patch
	args_lrs2["NORMALIZATION_MEAN"] = 0.4161 #mean value for normalization of greyscale lip region patch
	args_lrs2["NORMALIZATION_STD"] = 0.1688  #standard deviation value for normalization of greyscale lip region patch

	###########################################
	############### training ##################
	###########################################
	args_lrs2["SEED"] = 19220297 #seed for random number generators
	args_lrs2["BATCH_SIZE"] = 32 #minibatch size
	args_lrs2["EPOCH_SIZE"] = 16384   #number of samples in one step (virtual epoch)
	args_lrs2["NUM_EPOCHS"] = 1000 #maximum number of steps (Epochs) to train for (early stopping is used)

	###########################################
	################ model ####################
	###########################################
	args_lrs2["AUDIO_FEATURE_SIZE"] = 321    #feature size of audio features
	args_lrs2["NUM_CLASSES"] = 40    #number of output characters

	return args_lrs2


if __name__ == "__main__":

	args = get_LRS2_Cfg()

	for key,value in args.items():
		print(str(key) + " : " + str(value))
