
# TorchSC (IEEE-TGRS 2021): scene_classification_cnn_pytorch 

# Paper:

Enhanced Feature Pyramid Network with Deep Semantic Embedding for Remote Sensing Scene Classification

# Paper Link: 

https://ieeexplore.ieee.org/document/9314283/ 

# Usage:
```
$$ step 1
split images into train and test data;
generate txt file
python xx/split_train_test.py

$$ step 2 
train
python main.py

$$ step 3 
test
python predict.py

```

# Figs:
![Fig2](https://user-images.githubusercontent.com/85103981/120282868-293f8b00-c2ed-11eb-9b1a-6d3eb1510ce1.jpg)

![Fig5](https://user-images.githubusercontent.com/85103981/120283035-555b0c00-c2ed-11eb-9384-3d5f530834c6.jpg)


# Datasets:
UC Merced Land Use Dataset:

http://weegee.vision.ucmerced.edu/datasets/landuse.html

AID Dataset:

https://captain-whu.github.io/AID/


# Citation:
Please cite our paper if you find the work useful:
```
@ARTICLE{9314283,
  author={Wang, Xin and Wang, Shiyi and Ning, Chen and Zhou, Huiyu},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Enhanced Feature Pyramid Network With Deep Semantic Embedding for Remote Sensing Scene Classification}, 
  year={2021},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TGRS.2020.3044655}}
 ```
