# 3D-Graph-Convolutional-Networks-for-Medical-Image-denoising
This project, i.e.,3D GCN module, is for the paper "3D Graph Convolutional Networks for Medical Image Denoising.". We will upload this
paper to arxiv.com later. 
## Overall idea
You can impose this module to extract the non-local information of  medical image intra-slice with a play-and-plug fashion. Also, this can leverage the 3D spatial information between inter-slices. Finally, those information are aggregated as a hybrid one through a trainable weight.
## Versions
### 1)Play-and-Plug: 
   This can insert your backbone network directly with a manner of concatatenation. Then, the method of Fine-Tuning can be used to
   updata few of module parameters, which is not time-comsuming process. Finally, the extracted non-local and 3D spatial information consist of a        single feature map as the output of this module
### 2)General: 
   You MUST train the network from scratch. The output of this module is the way of 3D. 
## Framework
![images](https://github.com/tonyckc/3D-Graph-Convolutional-Networks-for-Medical-Image-denoising/tree/master/images/frame.png)
## Experimental results
1) ![images](https://github.com/tonyckc/3D-Graph-Convolutional-Networks-for-Medical-Image-denoising/blob/master/images/fig2.png)
we will add extra visualized results later. 
## Configrations
1) Python 3+;
2) tensorflow 1.12+
## End
if you have any questions, please feel free to contact me(cs.ckc96@gmail.com). 

