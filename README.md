# MPR-SNE
Source code for  "Multi-Path Relationship Preserved Social Network Embedding": https://ieeexplore.ieee.org/document/8649580

## Usage:
 1. Test node classification:
 `python main_node_classification.py`

 2. Test link prediction:
 `python main_link_prediction.py`

Note: the datasets should be placed in the datasets file, and the format refers to the datasets/BlogCataLog-dataset and the datasets/Flickr-dataset.

## Requirements
 - tensorflow>=1.2.1
 - Cython>=0.25.2
 - numpy>=1.14.2
 - scikit_learn>=0.20.3

## Installation
1. `pip install -r requirements.txt`
2. Building a Cython module using distutils:
`python  setup.py  build_ext  --inplace`. 
Refer to [http://docs.cython.org/en/latest/src/quickstart/build.html#building-a-cython-module-using-distutils](http://docs.cython.org/en/latest/src/quickstart/build.html#building-a-cython-module-using-distutils).

## Citing
If you find this code useful in your research, please cite the following paper:

> @article{lin2019multi,  
  title={Multi-Path Relationship Preserved Social Network Embedding},  
  author={Lin, Jianfeng and Zhang, Lei and He, Ming and Zhang, Hefu and Liu, Guiquan and Chen, Xiuyuan and Chen, Zhongming},  
  journal={IEEE Access},  
  volume={7},  
  pages={26507--26518},  
  year={2019},  
  publisher={IEEE}  
}
