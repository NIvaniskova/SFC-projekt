#!/bin/bash

# Install Python, pip, and venv
sudo apt update
sudo apt install -y python3 python3-pip python3-venv

# Create and activate environment
python3 -m venv ~/myenv
source ~/myenv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt