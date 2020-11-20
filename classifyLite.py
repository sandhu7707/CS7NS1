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

def decode(characters, y):
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

    with open(args.output, 'w') as output_file:
        interpreter = tflite.Interpreter(args.model_name)
        interpreter.allocate_tensors()
        inData = interpreter.get_input_details()
        outData = interpreter.get_output_details()

        for x in os.listdir(args.captcha_dir):
            # load image and preprocess it
            raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
            rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
            image = numpy.array(rgb_data) / 255.0
            (c, h, w) = image.shape
            image = image.reshape([-1, c, h, w])

            interpreter.set_tensor(inData[0]['index'], image)
            interpreter.invoke()

            prediction = []
            for output_node in outData:
                prediction.append(interpreter.get_tensor(output_node['index']))

            prediction = numpy.reshape(prediction, (len(outData),-1))
            res = decode(captcha_symbols, prediction)

            output_file.write(x + "," + res + "\n")
            print('Classified ' + x)

if __name__ == '__main__':
    main()