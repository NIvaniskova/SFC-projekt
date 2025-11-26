#!/bin/bash

# Install python3 and venv
sudo apt install python3
sudo apt install python3-pip
sudo apt install python3.12-venv

# Install and activate environment
python3 -m venv ~/myenv
source ~/myenv/bin/activate

# Install dependencies
pip3 install --upgrade pip
pip3 install packaging
pip3 install -r requirements.txt