import sys
import os
import argparse
#sys.path.append("/home/CONCURASP/kumara/helper_functions/")
#from fetch_image import FetchImage
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

import json
import argparse
from data_generator import gen_data
from model import OCR_Model
from plot_losses import TrainingPlot


if __name__ == "__main__":
    plot_losses_obj = TrainingPlot()
    print('Get arguments from user')
    parser = argparse.ArgumentParser()
    parser.add_argument('train_folder', help='folder containing the training images', type=str)
    parser.add_argument('train_file', help='annotation file for the validation data', type=str)
    parser.add_argument('val_folder', help='folder containing the validation images', type=str)
    parser.add_argument('val_file', help='annotation file for the validation data', type=str)
    parser.add_argument('vocab_file', help='vocab file', type=str)
    args = parser.parse_args()
    train_folder = os.path.join(args.train_folder, '')
    val_folder = os.path.join(args.val_folder, '')
    
    with open(args.vocab_file) as fp:
        rcpt_vocabulary = ['UNK', 'STOP','PAD'] + json.load(fp)
    rcpt_map = {}
    for i in range(len(rcpt_vocabulary)):
        rcpt_map[rcpt_vocabulary[i]] = i 
    
    char_num = len(rcpt_map)
    batch_size = 32
    
    weights = ModelCheckpoint('/home/CONCURASP/kumara/udacity_ocr_refactor/model/iteration_2/modelWeights-{epoch:03d}.h5', verbose=0, save_weights_only=True)
    callbacks_list = [weights,plot_losses_obj]

    print('Loading model file')
    model = OCR_Model(char_num).model   
    history = model.fit_generator(gen_data(train_folder, args.train_file,rcpt_map,batch_size),2500,epochs=50,   
                        validation_data=gen_data(val_folder, args.val_file,rcpt_map,batch_size),
                        validation_steps=500,callbacks=callbacks_list,
                        verbose=1)
    with open('/home/CONCURASP/kumara/udacity_ocr_refactor/model/iteration_2/history.json', 'w') as f:
        json.dump(history.history, f)
    model.save('/home/CONCURASP/kumara/udacity_ocr_refactor/model/iteration_2/ocr_extraction_model.h5', overwrite=True)