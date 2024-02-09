# FlexCare: Leveraging Cross-Task Synergy for Flexible Multimodal Healthcare Prediction
This is the source code for FlexCare: Leveraging Cross-Task Synergy for Flexible Multimodal Healthcare Prediction.

Requirements
----
This project was run in a conda virtual environment on Ubuntu 20.04 with CUDA 11.1. 
+ Pytorch==1.10.1
+ Python==3.7.9

Data preparation
----
In [mimic4extract](mimic4extract/)

Model training
----
python main_mt.py --task in-hospital-mortality,length-of-stay,decompensation,phenotyping,readmission,diagnosis --epochs 25 --lr 0.0001
