import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from augment import *
import sys
import json
import random




slice_height = 40
slice_width = 20
max_slice_num = 24
y_max_len = 40
max_img_height = 40

def get_text(vocab_file,pred_array):
    with open(vocab_file) as fp:
        rcpt_vocabulary = ['UNK', 'STOP','PAD'] + json.load(fp)
    ocr_text=''
    max_indices = np.argmax(pred_array, axis=2)
    
    for i in max_indices[0]:
        ocr_text+=rcpt_vocabulary[i]
    return ocr_text



def encode_text(text,rcpt_map):    
    output = np.zeros((y_max_len,len(rcpt_map)), dtype=np.bool)
    for i in range(len(text)):
        if text[i] in rcpt_map:
            char_pos = rcpt_map[text[i]]
        else:
            char_pos = rcpt_map['UNK']
        output[i, char_pos] = True 
        #print(text[i],output[i])
    output[i+1,rcpt_map['STOP']] = True
    for j in range(i+2,y_max_len):
        output[j,rcpt_map['PAD']] = True    
    return output
        
def get_slices(img):
    output = np.zeros((max_slice_num, slice_height, slice_width, 1))
    num_slices_img = int(img.shape[1]/slice_width)
    num_slices = min(num_slices_img,max_slice_num)
    for k in range(1, num_slices+1):
        output[k-1, :, :,0] = img[:,(k-1)*slice_width:k*slice_width]
    return output
def show_image(image):
    plt.imshow(image)
    plt.show()


def rgb2gray(rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    
def pre_process_image(image_path):
        try: 
            # open image
            im = Image.open(image_path)
            # get width and height of image
            img_width,img_height = im.size
            # get scale factor
            
            
            # rescale images to 40 times scale factor and convert to RGB channel numpy array
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

def get_image_path_text(input_folder,image_path):                    
    image_path = image_path[image_path.find(".")+2:]
    start_index = image_path[:image_path.rfind("_")].rfind("_")+1
    end_index = image_path.rfind("_")
    text = image_path[start_index:end_index]
    image_path = input_folder+image_path
    return (image_path,text)

def check_augment_image(augment_images_total,augment_image_count,img):
    augment_flag = bool(random.getrandbits(1))
    if (augment_flag)&(augment_image_count<augment_images_total):
        augment_image_count+=1
        img = augment_image(img)
    return (img,augment_image_count)

def gen_data(input_folder,annotation_file,rcpt_map,batch_size,augment_ratio=0.3,continuous=True):
    # Get number of characters in output
    char_num = len(rcpt_map)
    
    keep_going = True
    while keep_going:
        keep_going = continuous
        
        # Read in chunks
        chunks = pd.read_csv(annotation_file,sep=' ',names=['image_path','image_number'],chunksize=5000)
        for chunk in chunks:
            # shuffle dataframe
            chunk = chunk.sample(frac=1,random_state = 42).reset_index(drop=True)

            # number of images to be augmented
            augment_images_total = int(chunk.shape[0]*augment_ratio)
            augment_image_count = 0

            #print(augment_images_total)
            # for each chunk
            for chunk_batch in np.array_split(chunk,batch_size):
                # reset index
                chunk_batch = chunk_batch.reset_index(drop=True)
                # get number of rows
                number_of_samples = chunk_batch.shape[0]
                #print('number of rows in chunk',chunk.shape[0])
                #print('number_of_samples',number_of_samples)
                # define X and y arrays 
                X_train = np.zeros((number_of_samples, max_slice_num,slice_height, slice_width, 1))
                y_train = np.zeros((number_of_samples, y_max_len,char_num), dtype=np.bool)
                # For each row
                for index,row in chunk_batch.iterrows():
                    # get image_path,text
                    image_path,text = get_image_path_text(input_folder,row['image_path'])
                    print(image_path,text)
                    # pre-process image
                    img = pre_process_image(image_path)
                    #print(img.shape)
                    #show_image(img)

                    # augment image                                        
                    if img is not None:
                        img,augment_image_count = check_augment_image(augment_images_total,augment_image_count,img)
                        # split the image into slices by width
                        X_train[index, :, :, :, :] = get_slices(img)
                        y_train[index,:,:] = encode_text(text,rcpt_map)
                    else:
                        continue
                yield X_train,y_train