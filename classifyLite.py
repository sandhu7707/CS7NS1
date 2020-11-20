#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy
import string
import random
import argparse
import tflite_runtime.interpreter as tflite
import itertools

def decode(characters, y, len_y):
    y_idx = numpy.argmax(numpy.array(y), axis=1)
    y_pred = numpy.max(numpy.array(y), axis=1)
    res = ''.join([characters[x] for i,x in enumerate(y_idx) if y_pred[i]>0.5])
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the CNN model to use")
        exit(1)

    if args.captcha_dir is None:
         print("Please specify the directory with captchas to break")
         exit(1)

    if args.output is None:
        print("Please specify the path to the output file")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")

    if True:
        with open(args.output, 'w') as output_file:
            char_interpreter = tflite.Interpreter(args.model_name+'/model_cha.tflite')
            char_interpreter.allocate_tensors()

            char_input_d = char_interpreter.get_input_details()
            char_output_d = char_interpreter.get_output_details()

            for x in os.listdir(args.captcha_dir):
                # load image and preprocess it
                raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
                rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
                image = numpy.array(rgb_data, dtype=numpy.float32) / 255.0
                (c, h, w) = image.shape
                # assuming that input will have same size as of trained image
                image = image.reshape([-1, c, h, w])
                #predict from char-model
                char_interpreter.set_tensor(char_input_d[0]['index'], image)
                char_interpreter.invoke()
                prediction = []
                for output_node in char_output_d:
                    prediction.append(char_interpreter.get_tensor(output_node['index']))
                prediction = numpy.reshape(prediction, (len(char_output_d),-1))

                 #res = decode(captcha_symbols, prediction, len_prediction)
                res = decode_fix(captcha_symbols, prediction)
                output_file.write(x + "," + res + "\n")

                print('Classified ' + x)

if __name__ == '__main__':
    main()
