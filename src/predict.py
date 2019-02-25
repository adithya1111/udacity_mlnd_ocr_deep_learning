# Define the rnn ouput vocab
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%pylab inline
from PIL import Image
from model import OCR_Model
from collections import OrderedDict
from custom_recurrents import AttentionDecoder
from data_generator import *
import pandas as pd
import json
from keras.models import load_model

max_slice_num = 24
slice_width = 20
slice_height = 40

def get_end_char(string):
    if 'STOP' in string:
        end = 'STOP'
    else:
        end = 'PAD'
    return end

def get_text(vocab_file,pred_array):
    with open(vocab_file) as fp:
        rcpt_vocabulary = ['UNK', 'STOP','PAD'] + json.load(fp)
    ocr_text=''
    max_indices = np.argmax(pred_array, axis=2)
    
    for i in max_indices[0]:
        ocr_text+=rcpt_vocabulary[i]
    return ocr_text

#vocab_file = '/home/CONCURASP/kumara/udacity_ocr_refactor/src/rcptAlphabet.json'
def rgb2gray(rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
def pre_process_image(image_path):
    max_img_height = 40
    try:
        #print(image_path)
        im = Image.open(image_path)
        img_width,img_height = im.size
        scale_factor = int(img_width/float(img_height))
        if scale_factor <1:
            scale_factor =1
        total_width = scale_factor*40
        new_im = im.resize((total_width, max_img_height),Image.ANTIALIAS)        
        img_np = np.asarray(new_im)    
        img_np = np.resize(new_im, (max_img_height, total_width, 3))        
        img_np_grey = rgb2gray(img_np)
        return img_np_grey
        #print('Input shape',img_width,img_height)
    except:
        print("error reading image")
        return None
        #print(scale_factor)

# def load_model(vocab_file,model_file):
#     with open(vocab_file) as fp:
#         rcpt_vocabulary = ['UNK', 'STOP','PAD'] + json.load(fp)
#     rcpt_map = {}
#     for i in range(len(rcpt_vocabulary)):
#         rcpt_map[rcpt_vocabulary[i]] = i 
    
#     char_num = len(rcpt_map)
#     obj = OCR_Model(char_num,model_file)
#     return obj.model
def get_features(image_path):
           
    #show_image(img)
    img = pre_process_image(image_path)
    output = np.zeros((max_slice_num,slice_height, slice_width, 1))
    X_train = np.zeros((1, max_slice_num, slice_height, slice_width, 1))
    if img is not None:                        
        num_slices_img = int(img.shape[1]/slice_width)
        num_slices = min(num_slices_img,max_slice_num)
        #print('number of slices',num_slices)
        for k in range(1, num_slices+1):
            output[k-1, :, :,0] = img[:,(k-1)*slice_width:k*slice_width]
    X_train[0, :, :, :, :] = output  
    return X_train
def predict(X_train,vocab_file,model):
    pred = model.predict(X_train)    
    pred_string = get_text(vocab_file,pred)
    pred_string = pred_string[:pred_string.find(get_end_char(pred_string))]
    return pred_string

def call_predict(vocab_file,model_file,image_path):
    #model = load_model(vocab_file,model_file)
    

    model = load_model(model_file,custom_objects={'AttentionDecoder': AttentionDecoder})
    features = get_features(image_path)
    text = predict(features,vocab_file,model)
    print(text)
    return text

# base_path='/home/CONCURASP/kumara/udacity_ocr_refactor/data/ramdisk/max/90kDICT32px/'
# path = '1/1/100_Classmates_13991.jpg' 
# final_path = base_path+path
# vocab_file = '/home/CONCURASP/kumara/udacity_ocr_refactor/src/rcptAlphabet.json'
# model_file = '/home/CONCURASP/kumara/udacity_ocr_refactor/model/modelWeights-010.h5'
# call_predict(vocab_file,model_file,final_path)
