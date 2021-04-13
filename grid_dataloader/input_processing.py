#Load the video, process frame-by-frame, then save as .npy array

import numpy as np

from os import listdir
import os
from os.path import isfile, join
import torch
from tqdm import tqdm
import torch.nn as nn
import torch
import os
import cv2 as cv
from scipy.io import wavfile
from tqdm import tqdm
from src.data.lrs2_config import get_LRS2_Cfg
from src.models.deep_avsr.visual_frontend import VisualFrontend

from matplotlib import pyplot as plt

import cv2


#Function from: https://www.datacamp.com/community/tutorials/face-detection-python-opencv
haar_cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
def detect_faces(cascade, test_image, scaleFactor = 1.2):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()

    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)

    """for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)

    plt.imshow(image_copy)
    plt.show()"""

    faces_encountered = True
    if(len(faces_rect)==1):
        [[x,y,w,h]]=faces_rect
    elif(len(faces_rect)==0):
        faces_encountered = False
        print("*********************")
        print("NO FACES ENCOUNTERED")
        print("*********************")

        return 0, faces_encountered
    else:
        try:
            [x,y,w,h]=faces_rect[-1]
            """print("*********************")
            print("TWO RECTS+ ENCOUNTERED")
            print("*********************")

            cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
            plt.imshow(image_copy)
            plt.show()"""
            #import pdb; pdb.set_trace()
        except:
            import pdb; pdb.set_trace()

    image_copy = image_copy[y:y+h,x:x+w]

    return image_copy, faces_encountered


#DATA_GROUP = "s1"
NUM_FRAMES = 75
WINDOW_HEIGHT = 40
WINDOW_WIDTH = 40
#FRAME_HEIGHT = 288
#FRAME_WIDTH = 360

DATA_GROUPS = ["s1","s2","s3","s4","s5","s6","s7","s8","s9"]


for DATA_GROUP in DATA_GROUPS:

    all_files = [f for f in listdir('./grid_dataloader/GRID_DATA/'+DATA_GROUP) if isfile(join('./grid_dataloader/GRID_DATA/'+DATA_GROUP, f))]
    all_data =[]
    mpg_files_list = []

    for i in range(len(all_files)):
        if '.mpg' in all_files[i]:
            mpg_files_list.append(all_files[i])

    for i in tqdm(range(len(mpg_files_list))):
        if(i%100==0):
            print("%i / %i seen"%(i, len(mpg_files_list)))

        #try:

        videoFile = './grid_dataloader/GRID_DATA/'+DATA_GROUP+'/'+mpg_files_list[i]
        audioFile = './grid_dataloader/GRID_DATA/'+DATA_GROUP+'/inputs/input_'+mpg_files_list[i][:-4]+'.wav'
        v2aCommand = "ffmpeg -y -v quiet -i " + videoFile + " -ac 1 -ar 16000 -vn " + audioFile
        os.system(v2aCommand)

        vidcap = cv2.VideoCapture('./grid_dataloader/GRID_DATA/'+DATA_GROUP+'/'+mpg_files_list[i])

        success = True
        count = 0

        curr_arr = np.zeros((NUM_FRAMES,WINDOW_HEIGHT,WINDOW_WIDTH))

        while success:

            #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file

            success,image = vidcap.read()

            if(not(success)):
                #print('Read a new frame: ', success)
                break

            #print('Read a new frame: ', success)

            face,faces_encountered=detect_faces(haar_cascade_face,image)

            if(faces_encountered):

                lab_face = cv2.cvtColor(face,cv2.COLOR_BGR2LAB)


                (frame_height, frame_width, _) = lab_face.shape
                MU_X = 0
                #The Gaussian is centered at a 30% height of the image. This means it is
                #a mean of 0.2*(number of pixels in height) BELOW the center height

                #height parameters used for each of the speakers:
                #s1: 0.2
                #s2: 0.25
                #s3: 0.25
                #s4: 0.25
                #s5: 0.25
                #s6: 0.3
                #s7: 0.3
                #s8: 0.3
                #s9: 0.25

                MU_Y = 0.25*frame_height
                STD_DEV_X = STD_DEV_Y = 70

                x, y = np.meshgrid(np.linspace(-frame_width/2,frame_width/2,frame_width),
                                 np.linspace(-frame_height/2,frame_height/2,frame_height))

                g = (1/(2*np.pi*STD_DEV_X*STD_DEV_Y))*np.exp(- ( (x-MU_X)**2/(2*STD_DEV_X**2) + (y-MU_Y)**2/(2*STD_DEV_Y**2) ) )

                gauss_a_pixels = lab_face[:,:,1]*g/np.max(lab_face[:,:,1]*g)

                lip_pixels = np.argwhere(gauss_a_pixels>0.94)

                [center_y,center_x]=np.sum(lip_pixels,axis=0)/lip_pixels.shape[0]
                center_y,center_x=int(round(center_y)),int(round(center_x))

                """plt.imshow(np.where(gauss_a_pixels>0.94,gauss_a_pixels,0))
                plt.show()
                import pdb; pdb.set_trace()"""


                curr_mouth = face[center_y-20:center_y+20,center_x-20:center_x+20]

                """plt.imshow(curr_mouth)
                plt.show()"""
                #import pdb; pdb.set_trace()


                #Check and repair mouth being selected too low
                if(curr_mouth.shape[0]!=WINDOW_HEIGHT):
                    subtract_amount = WINDOW_HEIGHT - curr_mouth.shape[0]
                    curr_mouth = face[center_y-20-subtract_amount:center_y+20-subtract_amount,center_x-20:center_x+20]


                curr_mouth = cv2.cvtColor(curr_mouth,cv2.COLOR_BGR2GRAY)


                """plt.imshow(curr_mouth)
                plt.show()"""

                #Update the current frame with the normalized current mouth extracted
                try:
                    curr_arr[count]=(curr_mouth-np.min(curr_mouth))/(np.max(curr_mouth)-np.min(curr_mouth))
                except:
                    print("ERROR IN SELECTING AREA FOR MOUTH")
                    # import pdb; pdb.set_trace()

            #If face not present in this frame
            else:
                curr_arr[count] = np.zeros((WINDOW_HEIGHT,WINDOW_WIDTH))

            count += 1

            #if(count==70):
            #    import pdb; pdb.set_trace()


        # curr_arr would be 75, 40 , 40

        upsample = nn.Upsample(size=[112, 112], mode='bilinear')
        curr_arr = np.expand_dims(curr_arr, 1) # 75, 1, 40, 40
        curr_arr = upsample(torch.from_numpy(curr_arr))  # 75, 1, 112, 112

        np.random.seed(1234)
        torch.manual_seed(1234)
        gpuAvailable = torch.cuda.is_available()
        device = torch.device("cuda" if gpuAvailable else "cpu")

        # declaring the visual frontend module
        vf = VisualFrontend()
        vf.load_state_dict(torch.load('models/pre-trained_models/deep_avsr_visual_frontend.pt', map_location=device))
        vf.to(device)

        curr_arr = np.expand_dims(curr_arr, 0) # 1, 75, 1, 112, 112
        inputBatch = torch.from_numpy(curr_arr)
        inputBatch = (inputBatch.float()).to(device)
        vf.eval()
        with torch.no_grad():
            outputBatch = vf(inputBatch) # 1 , 75, 512
        out = torch.squeeze(outputBatch, axis=1) # 75, 512
        out = out.cpu().numpy()
        np.save('./grid_dataloader/GRID_DATA/' + DATA_GROUP + '/inputs/input_' + mpg_files_list[i][:-4] + '.npy', out)

