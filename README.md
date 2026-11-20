<div align="center">

# Panoramic Out-of-Distribution Segmentation

<p>
  <a href="https://github.com/MengfeiD"><strong>Mengfei Duan</strong></a>,
  <a href="https://github.com/7uHeng"><strong>Yuheng Zhang</strong></a>,
  <a href="https://scholar.google.com/citations?user=j3364z8AAAAJ"><strong>Yihong Cao</strong></a>,
  <a href="https://github.com/FeiT-FeiTeng"><strong>Fei Teng</strong></a>,
  <a href="https://github.com/xifen523"><strong>Kai Luo</strong></a>,
  <a href="https://jamycheung.github.io/"><strong>Jiaming Zhang</strong></a>,
  <a href="https://yangkailun.com/"><strong>Kailun Yang</strong></a><sup>†</sup>,
  <a href="http://robotics.hnu.edu.cn/info/1176/2960.htm"><strong>Zhiyong Li</strong></a><sup>†</sup>
</p>

<p>
  <a href="https://arxiv.org/abs/2505.03539">
    <img src="https://img.shields.io/badge/arXiv-2505.03539-b31b1b.svg?logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  <a href="https://arxiv.org/pdf/2505.03539">
    <img src="https://img.shields.io/badge/PDF-Download-blue.svg?logo=adobeacrobatreader&logoColor=white" alt="PDF">
  </a>
</p>

</div>

---

<p align="center">
  <img src="assets/panOoS.png" width="95%" alt="Overview of the PanOoS task">
</p>

This repository provides:

* **PanOoS**, a panoramic out-of-distribution segmentation task and evaluation protocol;
* **POS**, a prompt distribution learning framework designed for PanOoS;
* **DenseOoS**, a benchmark containing diverse outliers in complex panoramic environments;
* **QuadOoS**, a real-world benchmark captured by a quadruped robot equipped with a panoramic annular lens system.

---

## 📰 News

* **[2026-11]** Code and checkpoint are released.
* **[2026-10]** DenseOoS and QuadOoS are released.
* **[2026-06]** Environment installation and dataset preparation instructions released.
* **[2025-05]** Paper released on arXiv.
* **[2025-05]** Repository initialized.

---

## ✨ Highlights

* **New task.** We formulate Panoramic Out-of-Distribution Segmentation for comprehensive and safety-aware panoramic scene understanding.

* **New framework.** We propose **POS**, a prompt distribution learning framework that adapts vision-language representations to panoramic distortions and background clutter.

* **New benchmarks.** We establish **DenseOoS** and **QuadOoS** to support the development and evaluation of panoramic out-of-distribution segmentation.

* **Strong performance.** POS substantially improves panoramic out-of-distribution segmentation while preserving competitive closed-set semantic segmentation performance.

<p align="center">
  <img src="assets/overview.png" width="95%" alt="Overview of the proposed POS framework">
  <br>
  <em>Overview of the proposed POS framework.</em>
</p>

---

## 📊 Main Results

<p align="center">
  <strong>Panoramic out-of-distribution segmentation quantitative evaluation</strong>
</p>

<table align="center">
  <thead>
    <tr>
      <th rowspan="2" align="left">Method</th>
      <th colspan="3" align="center">DenseOoS</th>
      <th colspan="2" align="center">QuadOoS</th>
      <th colspan="2" align="center">Mean</th>
    </tr>
    <tr>
      <th align="center">AuPRC ↑</th>
      <th align="center">FPR<sub>95</sub> ↓</th>
      <th align="center">mIoU ↑</th>
      <th align="center">AuPRC ↑</th>
      <th align="center">FPR<sub>95</sub> ↓</th>
      <th align="center">AuPRC ↑</th>
      <th align="center">FPR<sub>95</sub> ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="8" align="left">
        <strong><em>Per-pixel methods</em></strong>
      </td>
    </tr>
    <tr>
      <td align="left">PEBAL</td>
      <td align="center">16.39</td>
      <td align="center">7.59</td>
      <td align="center">46.29</td>
      <td align="center">11.34</td>
      <td align="center"><u>39.25</u></td>
      <td align="center">13.87</td>
      <td align="center"><u>23.42</u></td>
    </tr>
    <tr>
      <td align="left">DenseHybrid</td>
      <td align="center">24.44</td>
      <td align="center">7.74</td>
      <td align="center">45.19</td>
      <td align="center">19.09</td>
      <td align="center"><strong>33.86</strong></td>
      <td align="center">21.77</td>
      <td align="center"><strong>20.80</strong></td>
    </tr>
    <tr>
      <td colspan="8" align="left">
        <strong><em>Mask-transformer-based methods</em></strong>
      </td>
    </tr>
    <tr>
      <td align="left">EAM</td>
      <td align="center">17.97</td>
      <td align="center"><u>6.46</u></td>
      <td align="center"><u>61.58</u></td>
      <td align="center">33.05</td>
      <td align="center">79.66</td>
      <td align="center">25.51</td>
      <td align="center">43.06</td>
    </tr>
    <tr>
      <td align="left">Mask2Anomaly</td>
      <td align="center">36.28</td>
      <td align="center">8.79</td>
      <td align="center">48.10</td>
      <td align="center">25.93</td>
      <td align="center">92.55</td>
      <td align="center">31.10</td>
      <td align="center">50.67</td>
    </tr>
    <tr>
      <td align="left">RbA</td>
      <td align="center"><u>51.31</u></td>
      <td align="center">21.87</td>
      <td align="center">57.54</td>
      <td align="center"><u>34.12</u></td>
      <td align="center">90.34</td>
      <td align="center"><u>42.71</u></td>
      <td align="center">56.11</td>
    </tr>
    <tr>
      <td align="left"><strong>POS (Ours)</strong></td>
      <td align="center"><strong>85.56</strong></td>
      <td align="center"><strong>0.45</strong></td>
      <td align="center"><strong>63.30</strong></td>
      <td align="center"><strong>48.13</strong></td>
      <td align="center">85.22</td>
      <td align="center"><strong>66.85</strong></td>
      <td align="center">42.84</td>
    </tr>
  </tbody>
  <tfoot>
    <tr>
      <td colspan="8" align="left">
        <sub>
          All values are reported in percentage points (%).
          Higher AuPRC and mIoU values are better, whereas lower
          FPR<sub>95</sub> values are better. The best and second-best
          results are highlighted in <strong>bold</strong> and
          <u>underlined</u>, respectively.
        </sub>
      </td>
    </tr>
  </tfoot>
</table>

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/MengfeiD/PanOoS.git
cd PanOoS
```

### 2. Install the Environment

Follow the complete installation instructions in
[`docs/INSTALL.md`](docs/INSTALL.md).

### 3. Prepare the Datasets

Follow [`docs/DATASET.md`](docs/DATASET.md) to download and organize
DenseOoS, QuadOoS, and the auxiliary training datasets.

### 4. Training, Fine-tuning and Evaluation
All commands in [`docs/RUN.md`](docs/RUN.md). Quick start:
```bash
# train (4 GPUs)
python train_net.py \
--config-file ckpts/clip_1dl/config.yaml \
--num-gpus 4 \
OUTPUT_DIR  model_logs/clip_1dl/

# fine-tuning
python train_net.py \
--config-file ckpts/clip_1dl_rba_ood_map_coco/config.yaml \
--num-gpus 4 \
OUTPUT_DIR  model_logs/clip_1dl_rba_ood_map_coco/

# evaluation
python evaluate_ood.py \
--out_path results/ \
--models_folder model_logs/ \
--datasets_folder PATH_TO_DATASETS_ROOT
```

---

## 📦 Weights

<table>
  <thead>
    <tr>
      <th rowspan="2" align="center">Model</th>
      <th colspan="3" align="center">DenseOoS</th>
      <th colspan="2" align="center">QuadOoS</th>
      <th rowspan="2" align="center">config</th>
      <th rowspan="2" align="center">ckpt</th>
    </tr>
    <tr>
      <th align="center">AuPRC</th>
      <th align="center">FPR<sub>95</sub></th>
      <th align="center">mIoU</th>
      <th align="center">AuPRC</th>
      <th align="center">FPR<sub>95</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">POS</td>
      <td align="center">85.56</td>
      <td align="center">0.45</td>
      <td align="center">63.30</td>
      <td align="center">48.13</td>
      <td align="center">85.22</td>
      <td align="center">
        <a href="ckpts/clip_1dl_rba_ood_map_coco/config.yaml">config</a>
      </td>
      <td align="center">
        <a href="https://drive.google.com/file/d/1SnJxVJYuRsqrL9IU2daMFfzPSxt_4ybB/view?usp=drive_link">model</a>
      </td>
    </tr>
  </tbody>
</table>

---

## 📚 Citation

If you find this work useful in your research, please consider citing the following paper:

```bibtex
@article{duan2025panoramic,
  title   = {Panoramic Out-of-Distribution Segmentation},
  author  = {Duan, Mengfei and Zhang, Yuheng and Cao, Yihong and Teng, Fei and Luo, Kai and Zhang, Jiaming and Yang, Kailun and Li, Zhiyong},
  journal = {IEEE Transactions on Circuits and Systems for Video Technology},
  year    = {2026},
  publisher = {IEEE}
}
```

---

## 🤝 Acknowledgments

This project builds upon several excellent open-source projects:

* [Mask2Former](https://github.com/facebookresearch/Mask2Former)
* [RbA](https://github.com/NazirNayal8/RbA)
* [FC-CLIP](https://github.com/bytedance/fc-clip)

We sincerely appreciate the tremendous efforts of their authors and contributors.