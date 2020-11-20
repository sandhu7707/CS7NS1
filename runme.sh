#!/bin/bash

#pulling from git

git clone https://github.com/sandhu7707/CS7NS1.git
cd CS7NS1

#virtual environment

python3 -m venv SINGHP1/
source SINGHP1/bin/activate
pip install opencv-python
pip install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp38-cp38-linux_armv7l.whl

#classify
python classifyLite.py --model-name var_len_classifier.tflite --captcha-dir SINGHP1-project2rpi/ --output newStuff.txt --symbols symbols.txt
