#!/bin/bash
#VM setup file 2

sudo fallocate -l 100G /mnt/100GB.swap
sudo mkswap /mnt/100GB.swap
sudo swapon /mnt/100GB.swap

conda update --all
cd ~/xgboost/python-package
python setup.py install
pip install tqdm
pip install keras
mkdir ~/bosch
mkdir ~/bosch/input
mkdir ~/bosch/scripts

#### if AWS ###
# sudo apt-get install awscli
# aws configure
# aws --region us-west-2 s3 cp s3://fredtonybosch/input/train_categorical.csv.zip ~/bosch/input
# aws --region us-west-2 s3 cp s3://fredtonybosch/input/test_categorical.csv.zip ~/bosch/input
# aws --region us-west-2 s3 cp s3://fredtonybosch/input/test_numeric.csv.zip ~/bosch/input
# aws --region us-west-2 s3 cp s3://fredtonybosch/input/train_numeric.csv.zip ~/bosch/input
# aws --region us-west-2 s3 cp s3://fredtonybosch/input/train_date.csv.zip ~/bosch/input
# aws --region us-west-2 s3 cp s3://fredtonybosch/input/test_date.csv.zip ~/bosch/input

### if google ###
cd ~/bosch/input
wget https://storage.googleapis.com/fredtonybosch/input/train_date.csv.zip
wget https://storage.googleapis.com/fredtonybosch/input/train_numeric.csv.zip
wget https://storage.googleapis.com/fredtonybosch/input/train_categorical.csv.zip
wget https://storage.googleapis.com/fredtonybosch/input/train_magic.csv.gz
wget https://storage.googleapis.com/fredtonybosch/input/test_numeric.csv.zip
wget https://storage.googleapis.com/fredtonybosch/input/test_date.csv.zip
wget https://storage.googleapis.com/fredtonybosch/input/test_categorical.csv.zip
wget https://storage.googleapis.com/fredtonybosch/input/sample_submission.csv.zip
wget https://storage.googleapis.com/fredtonybosch/input/test_magic.csv.gz

sudo apt-get install unzip
unzip test_categorical.csv.zip
unzip train_categorical.csv.zip
unzip test_numeric.csv.zip
unzip train_numeric.csv.zip
unzip train_date.csv.zip
unzip test_date.csv.zip
unzip sample_submission.csv.zip
gzip -d train_magic.csv.gz
gzip -d test_magic.csv.gz
rm *.zip
sed -i -e 's/T//g' train_categorical.csv
sed -i -e 's/T//g' test_categorical.csv

cd ~/bosch/scripts
wget https://storage.googleapis.com/fredtonybosch/scripts/cat_dupe.pkl
wget https://storage.googleapis.com/fredtonybosch/scripts/cat_non_dupe.pkl
wget https://storage.googleapis.com/fredtonybosch/scripts/num_dupe.pkl
wget https://storage.googleapis.com/fredtonybosch/scripts/num_non_dupe.pkl
wget https://storage.googleapis.com/fredtonybosch/scripts/date_dupe.pkl
wget https://storage.googleapis.com/fredtonybosch/scripts/date_non_dupe.pkl
wget https://storage.googleapis.com/fredtonybosch/scripts/train_date_mod.csv.gz
wget https://storage.googleapis.com/fredtonybosch/scripts/test_date_mod.csv.gz
gzip -d train_date_mod.csv.gz
gzip -d test_date_mod.csv.gz
#aws s3 cp cat_test_ohe_sp.npz s3://fredtonybosch/cat_test_ohe_sp.npz
#aws s3 cp cat_test_ohe_sp.npz s3://fredtonybosch/input/cat_test_ohe_sp.npz
echo Please upload Python scripts remotely
