# FlexCare
Source code for ***FlexCare: Leveraging Cross-Task Synergy for Flexible Multimodal Healthcare Prediction*** published in KDD 2024.

Requirements
----
This project was run in a conda virtual environment on Ubuntu 20.04 with CUDA 11.1. 
+ torch==1.10.1+cu111
+ Python==3.7.9
+ transformers==4.30.2
+ tokenizers==0.13.3
+ huggingface-hub==0.16.4

Data preparation
----
You will first need to request access for MIMIC dataset:
+ MIMIC-IV 2.0
+ MIMIC-CXR-JPG 2.0.0
+ MIMIC-IV-NOTE 2.2

Then follow the steps in [mimic4extract](mimic4extract/README.md) to build datasets for all tasks in directory [data].

In addition, we use _biobert-base-cased-v1.2_ as the pretrained text encoder, please download files in https://huggingface.co/dmis-lab/biobert-base-cased-v1.2, and put them into the directory [mymodel/pretrained]

Model training
----
``
python main_mt.py --data_path data --ehr_path data/ehr --cxr_path data/cxr --task in-hospital-mortality,length-of-stay,decompensation,phenotyping,readmission,diagnosis --epochs 25 --lr 0.0001 --device {gpu id} --seed 40
``
