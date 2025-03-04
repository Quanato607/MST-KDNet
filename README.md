>📋 For code accompanying Muti-scale Transformer Knowledge Distillation Network of the paper

# Bridging the Gap in Missing Modalities: Leveraging Knowledge Distillation and Style Matching for Brain Tumor Segmentation

<div align="center">

[![](https://img.shields.io/github/stars/Quanato607/MST-KDNet)](https://github.com/Quanato607/MST-KDNet)
[![](https://img.shields.io/github/forks/Quanato607/MST-KDNet)](https://github.com/Quanato607/MST-KDNet)
[![](https://img.shields.io/badge/project-page-red.svg)](https://github.com/Quanato607/MST-KDNet)
[![](https://img.shields.io/badge/arXiv-2403.01427-green.svg)](https://arxiv.org/abs/2030.12345)
</div>

This repository is the official implementation of **[MST-KDNet](https://arxiv.org/abs/2030.12345)**. Our method maintains **robust** and **accurate segmentation performance even under severe modality loss**. Furthermore, to reduce redundancy in modality-specific features, we incorporate **global and local feature refinements** to **systematically align available modalities and mitigate missing ones**.

## 🎥Visualization for Implementation on Software 

<div align="center">
<img src="https://github.com/Quanato607/MST-KDNet/blob/main/imgs/implementation.gif" width="90%">
</div>

## 💡Primary contributions

To overcome the challenges of missing or incomplete MRI modalities in brain tumor segmentation, we propose **MST-KDNet**. This is a novel framework for **cross-modality consistency** and **robust tumor segmentation in 3D medical images based on knowledge distillation and style matching**. Our key contributions are summarized as follows:

1) 🕐 MST-KDNet architecture achieves **efficient segmentation** under **missing modalities** by selectively aligning multi-scale Transformer features. This design effectively bridges modality gaps while preserving tumor boundary details.

2) 🕑 MST-KDNet significantly accelerates **inference**, **requiring only a compact distillation procedure instead of heavy fusion modules**, making it more adaptable to real-world clinical settings.

3) 🕒 We introduce **Global Style Matching Module (GSME)** to harmonize **heterogeneous modality features** and **retain texture consistency** even with severely missing imaging signals, without extra costly training data.

4) 🕓 Extensive experiments on both the **BraTS 2024** and **FeTS 2024 datasets** demonstrate **superior performance** and **robustness** of MST-KDNet, achieving state-of-the-art results especially in scenarios with multiple missing modalities.

## 🧗Proposed method
<br><br>
![](./imgs/fig1.png)
<br><br>

The overall framework of **MST-KDNet**. The Teacher propagation processes all available modalities, while the student propagation accommodates incomplete inputs.

## Table of Contents
- [Requirements](#-Requirements)
- [Training](#-Training)
- [Evaluation](#-Evaluation)
- [Results](#-Results)
- [Contributing](#-Contributing)

## 📝 Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## 🔥 Training

To train our model in the paper, run this command:

```train
python train.py
```

>📋 Before training, specify the data set and training configuration using the config.xml file

## 📃 Evaluation

To evaluate our model in the paper, run this command:

```eval
python eval.py
```

<br><br>
![](./imgs/fig2.png)
<br><br>

>📋 Comparison of segmentation results under four missing-modality scenarios: (1) all modalities, (2) FLAIR + T1ce + T2, (3) FLAIR + T1ce, and (4) FLAIR only. From left to right, the figure shows T1, T2, T1ce, and FLAIR images; ground-truth labels for two patients; three columns of comparison-study results; three columns of ablation-study results; and our final segmentation. Color legend: WT = red + yellow + green, TC = red + yellow, ET = red.
  
## 🚀 Results

Our model achieves the following performance on :

### [Comparison Experiment on BraTS 2024](https://www.synapse.org/Synapse:syn53708249)
<br><br>
![](./imgs/c1.png)
<br><br>
<br><br>
![](./imgs/c2.png)
<br><br>

### [Comparison Experiment on FeTS 2024](https://www.synapse.org/Synapse:syn53708249)
<br><br>
![](./imgs/c3.png)
<br><br>
<br><br>
![](./imgs/c4.png)
<br><br>

### Ablation Experiment on BraTS 2024 & FeTS 2024
<br><br>
![](./imgs/a1.png)
<br><br>

<table align="center" style="border-collapse: collapse; width:100%; text-align:center;">
  <thead>
    <tr>
      <!-- 第一行表头，Method 占两行，其余列分别合并三列Dice和三列HD95 -->
      <th rowspan="2" style="border:1px solid #000; padding:4px;">Method</th>
      <th colspan="3" style="border:1px solid #000; padding:4px;">Average Dice Score (%)</th>
      <th colspan="3" style="border:1px solid #000; padding:4px;">Average HD95 Score (mm)</th>
    </tr>
    <tr>
      <!-- 第二行表头：WT, TC, ET, 以及对应HD95的WT, TC, ET -->
      <th style="border:1px solid #000; padding:4px;">WT</th>
      <th style="border:1px solid #000; padding:4px;">TC</th>
      <th style="border:1px solid #000; padding:4px;">ET</th>
      <th style="border:1px solid #000; padding:4px;">WT</th>
      <th style="border:1px solid #000; padding:4px;">TC</th>
      <th style="border:1px solid #000; padding:4px;">ET</th>
    </tr>
  </thead>
  <tbody>
    <!-- 下面每行对应一组数据 -->
    <tr>
      <td style="border:1px solid #000; padding:4px;">RA-HVED</td>
      <td style="border:1px solid #000; padding:4px;">69.7</td>
      <td style="border:1px solid #000; padding:4px;">60.0</td>
      <td style="border:1px solid #000; padding:4px;">50.9</td>
      <td style="border:1px solid #000; padding:4px;">22.0</td>
      <td style="border:1px solid #000; padding:4px;">20.6</td>
      <td style="border:1px solid #000; padding:4px;">19.8</td>
    </tr>
    <tr>
      <td style="border:1px solid #000; padding:4px;">RMBTS</td>
      <td style="border:1px solid #000; padding:4px;">75.2</td>
      <td style="border:1px solid #000; padding:4px;">60.4</td>
      <td style="border:1px solid #000; padding:4px;">65.6</td>
      <td style="border:1px solid #000; padding:4px;">8.6</td>
      <td style="border:1px solid #000; padding:4px;">25.2</td>
      <td style="border:1px solid #000; padding:4px;">19.1</td>
    </tr>
    <tr>
      <td style="border:1px solid #000; padding:4px;">mmformer</td>
      <td style="border:1px solid #000; padding:4px;">68.9</td>
      <td style="border:1px solid #000; padding:4px;">54.6</td>
      <td style="border:1px solid #000; padding:4px;">48.6</td>
      <td style="border:1px solid #000; padding:4px;">26.7</td>
      <td style="border:1px solid #000; padding:4px;">27.5</td>
      <td style="border:1px solid #000; padding:4px;">34.0</td>
    </tr>
    <tr>
      <td style="border:1px solid #000; padding:4px;">M2FTrans</td>
      <td style="border:1px solid #000; padding:4px;">82.0</td>
      <td style="border:1px solid #000; padding:4px;">74.3</td>
      <td style="border:1px solid #000; padding:4px;">63.0</td>
      <td style="border:1px solid #000; padding:4px;">26.5</td>
      <td style="border:1px solid #000; padding:4px;">14.8</td>
      <td style="border:1px solid #000; padding:4px;">20.8</td>
    </tr>
    <tr>
      <td style="border:1px solid #000; padding:4px;">ACN</td>
      <td style="border:1px solid #000; padding:4px;">84.9</td>
      <td style="border:1px solid #000; padding:4px;">78.8</td>
      <td style="border:1px solid #000; padding:4px;">67.3</td>
      <td style="border:1px solid #000; padding:4px;">8.5</td>
      <td style="border:1px solid #000; padding:4px;">8.4</td>
      <td style="border:1px solid #000; padding:4px;">16.5</td>
    </tr>
    <tr>
      <td style="border:1px solid #000; padding:4px;">SMUNet</td>
      <td style="border:1px solid #000; padding:4px;"><span style="color:blue;">87.5</span></td>
      <td style="border:1px solid #000; padding:4px;"><span style="color:blue;">82.9</span></td>
      <td style="border:1px solid #000; padding:4px;"><span style="color:blue;">72.1</span></td>
      <td style="border:1px solid #000; padding:4px;"><span style="color:blue;">6.4</span></td>
      <td style="border:1px solid #000; padding:4px;"><span style="color:blue;">6.3</span></td>
      <td style="border:1px solid #000; padding:4px;"><span style="color:blue;">5.5</span></td>
    </tr>
    <tr>
      <td style="border:1px solid #000; padding:4px;">MST-KDNet</td>
      <td style="border:1px solid #000; padding:4px;"><span style="color:red;">88.4</span></td>
      <td style="border:1px solid #000; padding:4px;"><span style="color:red;">84.3</span></td>
      <td style="border:1px solid #000; padding:4px;"><span style="color:red;">73.4</span></td>
      <td style="border:1px solid #000; padding:4px;"><span style="color:red;">5.9</span></td>
      <td style="border:1px solid #000; padding:4px;"><span style="color:red;">5.7</span></td>
      <td style="border:1px solid #000; padding:4px;"><span style="color:red;">5.4</span></td>
    </tr>
  </tbody>
</table>


## 🤝 Contributing

>📋 Pick a licence and describe how to contribute to your code repository. 
