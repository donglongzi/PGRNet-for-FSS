# PGRNet-for-FSS

![image](./overview.png)

### Abstract
Few-shot semantic segmentation (FSS) is of tremendous potential for data-scarce scenarios, particularly in medical segmentation tasks with merely a few labeled data. Most of the existing FSS methods typically distinguish query objects with the guidance of support prototypes. However, the variances in appearance and scale between support and query objects from the same anatomical class are often exceedingly considerable in practical clinical scenarios, thus resulting in undesirable query segmentation masks. To tackle the aforementioned challenge, we propose a novel prototype-guided graph reasoning network (PGRNet) to explicitly explore potential contextual relationships in structured query images. Specifically, a prototype-guided graph reasoning module is proposed to perform information interaction on the query graph under the guidance of support prototypes to fully exploit the structural properties of query images to overcome intra-class variances. Moreover, instead of fixed support prototypes, a dynamic prototype generation mechanism is devised to yield a collection of dynamic support prototypes by mining rich contextual information from support images to further boost the efficiency of information interaction between support and query branches. Equipped with the proposed two components, PGRNet can learn abundant contextual representations for query images and is therefore more resilient to object variations. We validate our method on three publicly available medical segmentation datasets, namely CHAOS-T2, MS-CMRSeg, and Synapse. Experiments indicate that the proposed PGRNet outperforms previous FSS methods by a considerable margin and establishes a new state-of-the-art performance.

### Dependencies
Please install following essential dependencies:
```
dcm2nii
json5==0.8.5
jupyter==1.0.0
nibabel==2.5.1
numpy==1.22.0
opencv-python==4.5.5.62
Pillow>=8.1.1
sacred==0.8.2
scikit-image==0.18.3
SimpleITK==1.2.3
torch==1.10.2
torchvision=0.11.2
tqdm==4.62.3
```

### Datasets and pre-processing
Download:  
1) **CHAOS-MRI**: [Combined Healthy Abdominal Organ Segmentation data set](https://chaos.grand-challenge.org/)
2) **Synapse-CT**: [Multi-Atlas Abdomen Labeling Challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/218292)
3) **CMR**: [Multi-sequence Cardiac MRI Segmentation data set](https://zmiclab.github.io/projects/mscmrseg19/) (bSSFP fold)

**Pre-processing** is performed according to [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation/tree/2f2a22b74890cb9ad5e56ac234ea02b9f1c7a535) and we follow the procedure on their github repository.

**Supervoxel segmentation** is performed according to [Hansen et al.](https://github.com/sha168/ADNet.git) and we follow the procedure on their github repository.  
We also put the package `supervoxels` in `./data`, run our modified file `./data./supervoxels/generate_supervoxels.py` to implement pseudolabel generation. The generated supervoxels for `CHAOST2` and `CMR` datasets are put in `./data/CHAOST2/supervoxels_5000` folder and `./data/CMR/supervoxels_1000` folder, respectively.  

### Training  
1. Compile `./data/supervoxels/felzenszwalb_3d_cy.pyx` with cython (`python ./data/supervoxels/setup.py build_ext --inplace`) and run `./data/supervoxels/generate_supervoxels.py` 
2. Download pre-trained ResNet-101 weights [vanilla version](https://download.pytorch.org/models/resnet101-63fe2227.pth) or [deeplabv3 version](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth) and put your checkpoints folder, then replace the absolute path in the code `./models/encoder.py`.  
3. Run `./scripts/train_<abd,cmr>_mr.sh`  


### Testing
Run `./scripts/val.sh`

### Acknowledgement
This code is based on [SSL-ALPNet](https://arxiv.org/abs/2007.09886v2) (ECCV'20) by [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation.git) and [ADNet](https://www.sciencedirect.com/science/article/pii/S1361841522000378) by [Hansen et al.](https://github.com/sha168/ADNet.git). 

### Citation
```
@ARTICLE{huang2024prototype,
  author={Huang, Wendong and Hu, Jinwu and Xiao, Junhao and Wei, Yang and Bi, Xiuli and Xiao, Bin},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Prototype-Guided Graph Reasoning Network for Few-Shot Medical Image Segmentation}, 
  year={2024},
  volume={},
  number={},
  pages={1-1}
  doi={10.1109/TMI.2024.3459943}}
```
