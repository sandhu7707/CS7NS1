#!/bin/bash

#virtual environment

python3 -m venv env_singhp1/
source env_singhp1/bin/activate
pip install opencv-python
pip install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp38-cp38-linux_armv7l.whl

#classify
python classifyLite.py --model-name var_len_classifier.tflite --captcha-dir SINGHP1-project2rpi/ --output newStuff.txt --symbols symbols.txt
