#!/bin/bash

#virtual environment

python3 -m venv env/
source env/bin/activate
pip install scikit-build
pip install opencv-python
pip install tf-lite

python3 classify_.py --model-name var_len_classifier.tflite --captcha-dir SINGHP1-project2rpi/ --output stuff.txt --symbols symbols.txt
