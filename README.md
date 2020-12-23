# defocus-net
Official PyTorch implementation of [Focus on Defocus: Bridging the Synthetic to Real Domain Gap for Depth Estimation](https://openaccess.thecvf.com/content_CVPR_2020/html/Maximov_Focus_on_Defocus_Bridging_the_Synthetic_to_Real_Domain_Gap_CVPR_2020_paper.html) paper published at Conference on Computer Vision and Pattern Recognition (CVPR) 2020.

## Installation

Please download the code:

To use our code, first download the repository:
````
git clone https://github.com/dvl-tum/defocus-net.git
````

To install the dependencies:

````
pip install -r requirements.txt
````

## Training & Data

In order to train, run the following command:

````
python run_training.py
````

You can find the dataset [here](https://drive.google.com/file/d/1bR-WZQf44s0nsScC27HiEwaXPyEQ3-Dw/view?usp=sharing).

You would need to upload it to the 'data/' folder. The first 80% of it is the training data (400 samples). Each sample consists of a focal stack with 5 images and a depth file.

Data settings:
````
Focus distances: [0.1, 0.15, 0.3, 0.7, 1.5]. 
depth_max = 3 (meters)
focal_length = 2.9 * 1e-3
f_number = 1.
````

If you would like to render your own dataset, we provide a blender file with a python script to render focal stacks with depths (data/scene_focalstack.blend).
You would need to prepare object meshes and environment maps (+ optionally textures ).

Objects meshes can be downloaded from Thingi10k dataset:
https://ten-thousand-models.appspot.com/

Environment maps can be found at https://hdrihaven.com/



## Citation

If you find this code useful, please consider citing the following paper:

````
@InProceedings{Defocus_2020_CVPR,
author = {Maximov, Maxim and Galim, Kevin and Leal-Taixe, Laura},
title = {Focus on Defocus: Bridging the Synthetic to Real Domain Gap for Depth Estimation},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
````
