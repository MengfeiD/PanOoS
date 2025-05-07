<p align="center">

  <h1 align="center">Panoramic Out-of-Distribution Segmentation</h1>
  <p align="center">
    <a href="https://github.com/MengfeiD"><strong>Mengfei Duan</strong></a>
    .
    <a href="https://yangkailun.com/"><strong>Kailun Yang†</strong></a>
    .
    <a href=""><strong>Yuheng Zhang</strong></a>
    .
    <a href="https://scholar.google.com/citations?user=j3364z8AAAAJ"><strong>Yihong Cao</strong></a>
    .
    <a href=""><strong>Fei Teng</strong></a>
    .
    <a href="https://github.com/xifen523"><strong>Kai Luo</strong></a>
    .
    <a href="https://github.com/jamycheung"><strong>Jiaming Zhang</strong></a>
    .
    <a href="http://robotics.hnu.edu.cn/info/1071/1515.htm"><strong>Zhiyong Li†</strong></a>
    .
    <a href="https://robotics.hnu.edu.cn/info/1071/1514.htm"><strong>Shutao Li</strong></a>
</p>

<div align="center">
  
[`arXiv`](https://arxiv.org/pdf/2505.03539)
</div>

## PanOoS
<div align=center>
<img src="https://github.com/MengfeiD/PanOoS/blob/main/panOoS.png" >
</div>

## Framework
<div align=center>
<img src="https://github.com/MengfeiD/PanOoS/blob/main/overview.png" >
</div>

### Abstract
Panoramic imaging enables capturing 360° images with an ultra-wide Field-of-View (FoV) for dense omnidirectional perception. However, current panoramic semantic segmentation methods fail to identify outliers, and pinhole Out-of-distribution Segmentation (OoS) models perform unsatisfactorily in the panoramic domain due to background clutter and pixel distortions. To address these issues, we introduce a new task, Panoramic Out-of-distribution Segmentation (PanOoS), achieving OoS for panoramas. Furthermore, we propose the first solution, POS, which adapts to the characteristics of panoramic images through text-guided prompt distribution learning. Specifically, POS integrates a disentanglement strategy designed to materialize the cross-domain generalization capability of CLIP. The proposed Prompt-based Restoration Attention (PRA) optimizes semantic decoding by prompt guidance and self-adaptive correction, while Bilevel Prompt Distribution Learning (BPDL) refines the manifold of per-pixel mask embeddings via semantic prototype supervision. Besides, to compensate for the scarcity of PanOoS datasets, we establish two benchmarks: DenseOoS, which features diverse outliers in complex environments, and QuadOoS, captured by a quadruped robot with a panoramic annular lens system. Extensive experiments demonstrate superior performance of POS, with AuPRC improving by 34.25% and FPR95 decreasing by 21.42% on DenseOoS, outperforming state-of-the-art pinhole-OoS methods. Moreover, POS achieves leading closed-set segmentation capabilities.

## Updates
- [2025/05] Init repository. The code and datasets will be made publicly available upon acceptance of the paper. Thank you for your interest in our work!

## References
We appreciate the previous open-source works.
* [Mask2Former](https://github.com/facebookresearch/Mask2Former)
* [RbA](https://github.com/NazirNayal8/RbA)
* [FC-CLIP](https://github.com/bytedance/fc-clip)

## <a name="Citation"></a>Citation
If you are interested in this work, please use the following BibTeX entry.

```
@article{duan2025panoramic,
  title={Panoramic Out-of-Distribution Segmentation},
  author={Duan, Mengfei and Yang, Kailun and Zhang, Yuheng and Cao, Yihong and Teng, Fei and Luo, Kai and Zhang, Jiaming and Li, Zhiyong and Li, Shutao},
  journal={arXiv preprint arXiv:2505.03539},
  year={2025}
}
```
