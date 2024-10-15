# PGRNet-for-FSS

![image](./overview.png)

### Abstract
Few-shot semantic segmentation (FSS) is of tremendous potential for data-scarce scenarios, particularly in medical segmentation tasks with merely a few labeled data. Most of the existing FSS methods typically distinguish query objects with the guidance of support prototypes. However, the variances in appearance and scale between support and query objects from the same anatomical class are often exceedingly considerable in practical clinical scenarios, thus resulting in undesirable query segmentation masks. To tackle the aforementioned challenge, we propose a novel prototype-guided graph reasoning network (PGRNet) to explicitly explore potential contextual relationships in structured query images. Specifically, a prototype-guided graph reasoning module is proposed to perform information interaction on the query graph under the guidance of support prototypes to fully exploit the structural properties of query images to overcome intra-class variances. Moreover, instead of fixed support prototypes, a dynamic prototype generation mechanism is devised to yield a collection of dynamic support prototypes by mining rich contextual information from support images to further boost the efficiency of information interaction between support and query branches. Equipped with the proposed two components, PGRNet can learn abundant contextual representations for query images and is therefore more resilient to object variations. We validate our method on three publicly available medical segmentation datasets, namely CHAOS-T2, MS-CMRSeg, and Synapse. Experiments indicate that the proposed PGRNet outperforms previous FSS methods by a considerable margin and establishes a new state-of-the-art performance.


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
