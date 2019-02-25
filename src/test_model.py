# Define the rnn ouput vocab
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%pylab inline
from PIL import Image
from model import OCR_Model
from collections import OrderedDict
from predict import *
import pandas as pd
import json
import argparse

max_slice_num = 24
slice_width = 20
slice_height = 40

def get_image_path_text(image_path,input_folder):                    
    image_path = image_path[image_path.find(".")+2:]
    start_index = image_path[:image_path.rfind("_")].rfind("_")+1
    end_index = image_path.rfind("_")
    text = image_path[start_index:end_index]
    image_path = input_folder+image_path
    return image_path,text

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

def load_model(vocab_file,model_file):
    with open(vocab_file) as fp:
        rcpt_vocabulary = ['UNK', 'STOP','PAD'] + json.load(fp)
    rcpt_map = {}
    for i in range(len(rcpt_vocabulary)):
        rcpt_map[rcpt_vocabulary[i]] = i 
    
    char_num = len(rcpt_map)
    obj = OCR_Model(char_num,model_file)
    return obj.model
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
def predict(image_path,model,vocab_file):    
    features = get_features(image_path)
    pred = model.predict(features)    
    pred_string = get_text(vocab_file,pred)
    pred_string = pred_string[:pred_string.find(get_end_char(pred_string))]
    print(pred_string)
    return pred_string

def parse_test_file(vocab_file,model_file,test_file,input_folder):    
    model = load_model(vocab_file,model_file)
    chunks = pd.read_csv(test_file,sep=' ',names=['image_path','image_number'],chunksize=5000,nrows=10000)
    processed_images = 0
    correct_images = 0
    for chunk in chunks:
        for index,row in chunk.iterrows():
            image_path,true_string = get_image_path_text(row['image_path'],input_folder)
            pred_string = predict(image_path,model,vocab_file)
            processed_images+=1
            if true_string==pred_string:
                correct_images+=1
    print(str(correct_images) + ' correct of ' + str(processed_images) + ' total (' + '{0:.2f}'.format(100 * correct_images / processed_images) + '%)')

    
if __name__ == "__main__":
    print('Get arguments from user')
    parser = argparse.ArgumentParser()
    parser.add_argument('vocab_file', help='vocab_file', type=str)
    parser.add_argument('model_file', help='model_file', type=str)
    parser.add_argument('test_file', help='model_file', type=str)
    parser.add_argument('input_folder', help='model_file', type=str)
    args = parser.parse_args()
    parse_test_file(args.vocab_file,args.model_file,args.test_file,args.input_folder)
    #main(vocab_file,model_file,image_path)

# base_path='/home/CONCURASP/kumara/udacity_ocr_refactor/data/ramdisk/max/90kDICT32px/'
# path = '1/1/100_Classmates_13991.jpg' 
# final_path = base_path+path
# vocab_file = '/home/CONCURASP/kumara/udacity_ocr_refactor/src/rcptAlphabet.json'
# model_file = '/home/CONCURASP/kumara/udacity_ocr_refactor/model/modelWeights-010.h5'
# #call_main(vocab_file,model_file,final_path)
