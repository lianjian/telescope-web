#!/bin/bash
echo "Creating and activating venv"
python3 -m venv venv
. venv/bin/activate

echo "Installing dependencies to venv"
pip install -e .

echo "Downloading model files"
wget https://www.dropbox.com/s/9abcuj1jkxfr8kw/model.tar.gz
tar -zxvf model.tar.gz
mv model/ telescope_flask/

echo "Removing model.tar.gz"
rm model.tar.gz
