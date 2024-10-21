import sys
import numpy as np
import nnfs
import math
from PIL import Image
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import random
from multiprocessing import Pool, Process, Queue

images_tested = 0
totalsamples = 1106

one_hot_org = np.empty(30) # EXPECTED OUTPUTS

path = None
try:
    this_dir  = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    path = this_dir + r'\\train'
except FileNotFoundError:
    print("FILE DOESNT EXIST")

print(path)

def load_images_from_folder(folder):
    images = []
    folderlist = os.listdir(folder)

    for i in range(0,30): ## train n images at a time
        imgfolder = random.randint(0,totalsamples)-1
        ranimg = imgfolder*2 # take image of set
        txtfile = ranimg+1
        with open(os.path.join(folder,folderlist[txtfile])) as f:
            r = f.readlines()
            one_hot_org[i] = int(r[0])
            print(r)

        img = mpimg.imread(os.path.join(folder,folderlist[ranimg]))
        if img is not None:
            images.append(img)
    return images

## DATA
def load_imgs(one_hot):

    images_list = load_images_from_folder(path)

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], np.array([0.2989, 0.5870, 0.1140]))
    
    grays = []

    for image in images_list:
        gray = rgb2gray(image)
        grays.append(gray.transpose())


    X = grays

        ## SHUFFLING 
    shuffling_array = []
    for i in range(0,30):
        shuffling_array.append(i)
    shuffling_array = np.array(shuffling_array)
    np.random.shuffle(shuffling_array)

    for i in range(0,30):
        one_hot[i] = one_hot[shuffling_array[i]]
        grays[i] = grays[shuffling_array[i]]
    return X, one_hot


#plt.imshow(gray, cmap=plt.get_cmap('gray'))
#plt.show()


##WEIGHTS

## BEGNNING OF NEW NEURON IN PATH
#how best tune weights and bias to achieve desired output



class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs,n_neurons) ## weights
        self.biases = 0.10*np.random.randn(1,n_neurons)
    def forward(self, inputs):
        self.output = np.dot(inputs,self.weights) + self.biases


class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

class Convolutional_Filter:
    def __init__(self,width,height):
        self.weights = 0.10*np.random.randn(width,height) # weights
        self.bias = 0.10*np.random.randn(1,1)
        self.width = width
        self.height = height
    def forward(self,input):
        input = input.astype(np.float32)
        x = -1
        feature_map = [[]]
        
        for i in range(0,len(input)-1-self.width,14): # rows
            x+=1
            for j in range(0,(len(input[i])-1-self.height),14): # column
                section = input[i:i+self.width,j:j+self.height]
                convulsion = np.sum(np.inner(section,self.weights)) + self.bias
                convulsion = float(convulsion)
                feature_map[x].append(convulsion)
            feature_map.append([])
        feature_map.pop()
        npft_map = np.asarray(feature_map)
        self.output = npft_map

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        probabilities = exp_values / np.sum(exp_values,axis=1,keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self,output, y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,10^-7,(1-(10^-7)))



        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true.astype(int)]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true,axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

def filter2(input_array): # searches 4by4 grid
    gridsize = 4
    final_array = []
    for i in range(0,len(input_array)-1,gridsize):
        for j in range(0,(len(input_array[i])-1),gridsize):
            section = input_array[i:i+gridsize,j:j+gridsize]
            nmax = np.max(section)
            final_array.append(nmax)
    final_array = np.array(final_array)
    return final_array

def derive(y2,y1,dx):
    return (y2-y1)/dx


lossfx = Loss_CategoricalCrossentropy()



def initialize(layers):
    pass

## vars

denses = []
activations = []

##                               initialize layers #####

## FILTER REQUIREMENTS
filter1 = Convolutional_Filter(112,112)
activation1 = Activation_ReLU()

## backend layers
denses.append(Layer_Dense(4,32))
denses.append(Layer_Dense(32,64))
denses.append(Layer_Dense(64,128))
denses.append(Layer_Dense(128,256))
denses.append(Layer_Dense(256,128))
denses.append(Layer_Dense(128,64))
denses.append(Layer_Dense(64,2))

activations.append(Activation_ReLU())
activations.append(Activation_ReLU())
activations.append(Activation_ReLU())
activations.append(Activation_ReLU())
activations.append(Activation_ReLU())
activations.append(Activation_ReLU())
activations.append(Activation_Softmax())


precision = 0.01
## training functions



def run(input):

    for i in range(0,len(denses)):
        if i == 0:
            denses[i].forward(input)
        else:
            denses[i].forward(activations[i-1].output)
        activations[i].forward(denses[i].output)
    
    loss = lossfx.calculate(activations[len(denses)-1].output,one_hot)
    return loss

def run_s(prevout,layerum):
    denses[layerum].forward(prevout)
    for i in range(layerum, len(denses)):
        denses[i].forward(activations[i-1].output)
        activations[i].forward(denses[i].output)
    loss = lossfx.calculate(activations[len(denses)-1].output,one_hot)
    return loss


def adjust_val(medium,input, layernum, prevout):
        org_medium = medium.copy()
        orgloss = run(input)
        for h in range(0,len(medium)):
            for l in range(0,len(medium[h])):
                if layernum == 0:
                    loss1 = run(input)
                    medium[h][l]+=precision
                    loss2 = run(input)
                    medium[h][l]-=precision
                    diff = derive(loss2,loss1,precision) ### FIND SLOPE OF TANGENT HERE

                    original = medium[h][l]
                    #lose1 = run(input)
                    if diff > 0:
                        medium[h][l]-=precision
                    elif diff < 0:
                        medium[h][l]+=precision
                    '''lose2 = run(input)
                    if lose2 >= lose1:
                        medium[h][l] = original'''

                else:

                    loss1 = run_s(prevout,layernum)
                    medium[h][l]+=precision
                    loss2 = run_s(prevout,layernum)
                    medium[h][l]-=precision
                    diff = derive(loss2,loss1,precision) ### FIND SLOPE OF TANGENT HERE

                    #lose1 = run_s(prevout,layernum)
                    if diff > 0:
                        medium[h][l]-=precision
                    elif diff < 0:
                        medium[h][l]+=precision
                    '''lose2 = run_s(prevout,layernum)
                    if lose2 >= lose1:
                        medium[h][l] = original'''
            #CHANGE BIAs

        final_loss = run(input)
        if final_loss > orgloss:
            medium = org_medium


                




def adj_filter(medium, rawinput,orginput):
    org_medium = medium.copy()
    orgloss = run(orginput)
    for h in range(0,len(medium)):
        print(h / len(medium))
        for l in range(0,len(medium[h])):
###############################
            filter1.forward(rawinput)
            activation1.forward(filter1.output)
            org_input = filter2(activation1.output) ## NEW INPUT 9

            loss1 = run(org_input)
            medium[h][l]+=precision

            filter1.forward(rawinput)
            activation1.forward(filter1.output)
            new_input = filter2(activation1.output) ## NEW INPUT 9

            loss2 = run(new_input)
            medium[h][l]-=precision
            diff = derive(loss2,loss1,precision)
####################
            if diff > 0:
                medium[h][l]-=precision
            elif diff < 0:
                medium[h][l]+=precision
    final_loss = run(input)
    if final_loss > orgloss:
        medium = org_medium
            

last_loss = 0


losses = []
accuracies = []

def worker(data_img):

    ## ADJUST FILTER
    filter1.forward(data_img)
    activation1.forward(filter1.output)
    max_pool = filter2(activation1.output) ## NEW INPUT
    #print("ADJUSTING FILTER")
    #adj_filter(filter1.weights,data_img,max_pool)

    ## ADJUST BACKEND NEURAL NETWORK WEIGHTS FOR 9 INPUT
    #print("ADJUSTING WEIGHTS")
    for i in range(0,len(denses)):
        prevout = None
        if i != 0:
            prevout = activations[i-1].output
        adjust_val(denses[i].weights,max_pool,i,prevout)
        adjust_val(denses[i].biases,max_pool,i,prevout)
    losses.append(run(max_pool))
    softmax_out = activations[len(denses)-1].output
    predictions = np.argmax(softmax_out,axis=1)

    accuracy = np.mean(predictions == one_hot)
    accuracies.append(accuracy)
    print("accuracy is currently: ", accuracy)


while True:
    losses = []
    accuracies = []
    one_hot = one_hot_org
    # GEN NEW IMG
    X, one_hot = load_imgs(one_hot)

    

    iteration = -1
    ## BACKPROPOGATION
    for imagedata in X:
        worker(imagedata)
        iteration +=1
        print("Image ", iteration, "/30")
    losses = np.array(losses)
    print(np.mean(losses))
    
    print(np.mean(accuracies))
    images_tested+=30
    print(images_tested)
    




#### TRAINING ########################################################

