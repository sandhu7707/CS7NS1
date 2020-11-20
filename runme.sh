#!/bin/bash

#virtual environment

python3 -m venv env_singhp1/
source env_singhp1/bin/activate
pip install opencv-python
pip install tf-lite

python3 classify_.py --model-name var_len_classifier.tflite --captcha-dir SINGHP1-project2rpi/ --output stuff.txt --symbols symbols.txt
