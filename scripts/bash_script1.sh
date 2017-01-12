#!/bin/bash
#VM setup file 1

wget https://repo.continuum.io/archive/Anaconda2-4.2.0-Linux-x86_64.sh
bash Anaconda2-4.2.0-Linux-x86_64.sh 
rm Anaconda2-4.2.0-Linux-x86_64.sh
sudo apt-get install git
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
sudo apt-get install make
sudo apt-get update
sudo apt-get install g++
make -j4
sudo reboot
