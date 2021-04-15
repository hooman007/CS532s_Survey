import numpy as np
from os import listdir
from os.path import isfile, join

import grid_dictionaries as gd

for dataset_indx in range(1, 10):
    DATA_GROUP = f"s{dataset_indx}"

    #Vector shape: (6x51)
    grid_dics = [gd.command_dic,
                 gd.color_dic,
                 gd.preposition_dic,
                 gd.letter_dic,
                 gd.digit_dic,
                 gd.adverb_dic]

    all_files = [f for f in listdir('./GRID_DATA/'+DATA_GROUP) if isfile(join('./GRID_DATA/'+DATA_GROUP, f))]

    mpg_files_list = []

    for i in range(len(all_files)):
        if '.mpg' in all_files[i]:
            mpg_files_list.append(all_files[i])

    for i in range(len(mpg_files_list)):
        if(i%100==0):
            print("%i / %i seen"%(i, len(mpg_files_list)))

        try:
            curr_arr = np.zeros((6,51))
            #Iterate through the letter of the current file, excpet for '.mpg'
            for lett in range(len(mpg_files_list[i])-4):
                curr_arr[lett][grid_dics[lett][mpg_files_list[i][lett]]] = 1

            np.save("./GRID_DATA/"+DATA_GROUP+"/labels/"+mpg_files_list[i][:len(mpg_files_list[i])-4]+".npy",curr_arr)
            #import pdb; pdb.set_trace()
        except:
            print("\n")
            print("ERROR reading file:",str(i))
            print("\n")
