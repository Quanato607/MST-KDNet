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

<table style="border-collapse: collapse; width:100%; text-align:center;">
  <caption style="margin-bottom:6px;">
    Comparison of Dice score for various state-of-the-art models (BraTS 2024). 
    <span style="color:red;">Red</span> represents the optimal value, 
    and <span style="color:blue;">Blue</span> represents the suboptimal value.
  </caption>
  <!-- 表头部分 -->
  <thead>
    <!-- 第一行：Type 与 FLAIR/T1/T1Gd/T2 多行合并 -->
    <tr>
      <th rowspan="4" style="border:1px solid #000; padding:4px;">Type</th>
      <th rowspan="4" style="border:1px solid #000; padding:4px;">
        FLAIR<br>T1<br>T1Gd<br>T2
      </th>
      <!-- 下方是 16 列圆圈/实心圆符号（分成四行），以及最右的 Avg. 合并四行 -->
      <th style="border:1px solid #000; padding:4px;">&#9675;</th>
      <th style="border:1px solid #000; padding:4px;">&#9675;</th>
      <th style="border:1px solid #000; padding:4px;">&#9675;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;">&#9675;</th>
      <th style="border:1px solid #000; padding:4px;">&#9675;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;">&#9675;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;">&#9675;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th rowspan="4" style="border:1px solid #000; padding:4px;">Avg.</th>
    </tr>
    <!-- 第二行 -->
    <tr>
      <th style="border:1px solid #000; padding:4px;">&#9675;</th>
      <th style="border:1px solid #000; padding:4px;">&#9675;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;">&#9675;</th>
      <th style="border:1px solid #000; padding:4px;">&#9675;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;">&#9675;</th>
      <th style="border:1px solid #000; padding:4px;">&#9675;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;">&#9675;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
    </tr>
    <!-- 第三行 -->
    <tr>
      <th style="border:1px solid #000; padding:4px;">&#9675;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;">&#9675;</th>
      <th style="border:1px solid #000; padding:4px;">&#9675;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;">&#9675;</th>
      <th style="border:1px solid #000; padding:4px;">&#9675;</th>
      <th style="border:1px solid #000; padding:4px;">&#9675;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;">&#9675;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;"></th>
    </tr>
    <!-- 第四行 -->
    <tr>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;">&#9675;</th>
      <th style="border:1px solid #000; padding:4px;">&#9675;</th>
      <th style="border:1px solid #000; padding:4px;">&#9675;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;">&#9675;</th>
      <th style="border:1px solid #000; padding:4px;">&#9675;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;">&#9675;</th>
      <th style="border:1px solid #000; padding:4px;">&#9675;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;">&#9679;</th>
      <th style="border:1px solid #000; padding:4px;"></th>
    </tr>
  </thead>
  
  <!-- 表格内容部分 -->
  <tbody>
    <!-- WT 行数据 -->
    <tr>
      <td style="border:1px solid #000; padding:4px;" rowspan="7">WT</td>
      <td style="border:1px solid #000; padding:4px;">RA-HVED</td>
      <td style="border:1px solid #000; padding:4px;"><span style="color:blue;">75.4</span></td>
      <td style="border:1px solid #000; padding:4px;">51.3</td>
      <td style="border:1px solid #000; padding:4px;">9.5</td>
      <td style="border:1px solid #000; padding:4px;">71.4</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">77.5</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">53.4</td>
      <td style="border:1px solid #000; padding:4px;">72.9</td>
      <td style="border:1px solid #000; padding:4px;">76.1</td>
      <td style="border:1px solid #000; padding:4px;">80.1</td>
      <td style="border:1px solid #000; padding:4px;">72.9</td>
      <td style="border:1px solid #000; padding:4px;">72.9</td>
      <td style="border:1px solid #000; padding:4px;">80.6</td>
      <td style="border:1px solid #000; padding:4px;">80.4</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">77.7</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">80.1</td>
      <td style="border:1px solid #000; padding:4px;">68.8</td>
    </tr>
    <tr>
      <td style="border:1px solid #000; padding:4px;">RMBTS</td>
      <td style="border:1px solid #000; padding:4px;">70.1</td>
      <td style="border:1px solid #000; padding:4px;">51.2</td>
      <td style="border:1px solid #000; padding:4px;">51.8</td>
      <td style="border:1px solid #000; padding:4px;">65.0</td>
      <td style="border:1px solid #000; padding:4px;">75.3</td>
      <td style="border:1px solid #000; padding:4px;">60.6</td>
      <td style="border:1px solid #000; padding:4px;">76.4</td>
      <td style="border:1px solid #000; padding:4px;">75.0</td>
      <td style="border:1px solid #000; padding:4px;">77.3</td>
      <td style="border:1px solid #000; padding:4px;">76.0</td>
      <td style="border:1px solid #000; padding:4px;">79.3</td>
      <td style="border:1px solid #000; padding:4px;">79.7</td>
      <td style="border:1px solid #000; padding:4px;">80.3</td>
      <td style="border:1px solid #000; padding:4px;">76.1</td>
      <td style="border:1px solid #000; padding:4px;">80.9</td>
      <td style="border:1px solid #000; padding:4px;">71.7</td>
    </tr>
    <tr>
      <td style="border:1px solid #000; padding:4px;">mmformer</td>
      <td style="border:1px solid #000; padding:4px;">72.6</td>
      <td style="border:1px solid #000; padding:4px;">55.5</td>
      <td style="border:1px solid #000; padding:4px;">61.3</td>
      <td style="border:1px solid #000; padding:4px;">72.7</td>
      <td style="border:1px solid #000; padding:4px;">74.3</td>
      <td style="border:1px solid #000; padding:4px;">65.4</td>
      <td style="border:1px solid #000; padding:4px;">79.2</td>
      <td style="border:1px solid #000; padding:4px;">75.1</td>
      <td style="border:1px solid #000; padding:4px;">79.6</td>
      <td style="border:1px solid #000; padding:4px;">78.3</td>
      <td style="border:1px solid #000; padding:4px;">80.0</td>
      <td style="border:1px solid #000; padding:4px;">80.7</td>
      <td style="border:1px solid #000; padding:4px;">81.0</td>
      <td style="border:1px solid #000; padding:4px;">75.6</td>
      <td style="border:1px solid #000; padding:4px;">81.3</td>
      <td style="border:1px solid #000; padding:4px;">74.2</td>
    </tr>
    <tr>
      <td style="border:1px solid #000; padding:4px;">M2FTrans</td>
      <td style="border:1px solid #000; padding:4px;">72.5</td>
      <td style="border:1px solid #000; padding:4px;">58.8</td>
      <td style="border:1px solid #000; padding:4px;">62.0</td>
      <td style="border:1px solid #000; padding:4px;">73.0</td>
      <td style="border:1px solid #000; padding:4px;">73.9</td>
      <td style="border:1px solid #000; padding:4px;">64.2</td>
      <td style="border:1px solid #000; padding:4px;">77.4</td>
      <td style="border:1px solid #000; padding:4px;">73.6</td>
      <td style="border:1px solid #000; padding:4px;">78.9</td>
      <td style="border:1px solid #000; padding:4px;">77.0</td>
      <td style="border:1px solid #000; padding:4px;">77.3</td>
      <td style="border:1px solid #000; padding:4px;">78.5</td>
      <td style="border:1px solid #000; padding:4px;">79.5</td>
      <td style="border:1px solid #000; padding:4px;">74.2</td>
      <td style="border:1px solid #000; padding:4px;">78.8</td>
      <td style="border:1px solid #000; padding:4px;">73.3</td>
    </tr>
    <tr>
      <td style="border:1px solid #000; padding:4px;">ACN</td>
      <td style="border:1px solid #000; padding:4px;">69.6</td>
      <td style="border:1px solid #000; padding:4px;">58.7</td>
      <td style="border:1px solid #000; padding:4px;">60.1</td>
      <td style="border:1px solid #000; padding:4px;">80.7</td>
      <td style="border:1px solid #000; padding:4px;">71.8</td>
      <td style="border:1px solid #000; padding:4px;">63.6</td>
      <td style="border:1px solid #000; padding:4px;">82.1</td>
      <td style="border:1px solid #000; padding:4px;">72.2</td>
      <td style="border:1px solid #000; padding:4px;">82.3</td>
      <td style="border:1px solid #000; padding:4px;">81.3</td>
      <td style="border:1px solid #000; padding:4px;">82.0</td>
      <td style="border:1px solid #000; padding:4px;">82.8</td>
      <td style="border:1px solid #000; padding:4px;">82.0</td>
      <td style="border:1px solid #000; padding:4px;">72.5</td>
      <td style="border:1px solid #000; padding:4px;">82.5</td>
      <td style="border:1px solid #000; padding:4px;">75.0</td>
    </tr>
    <tr>
      <td style="border:1px solid #000; padding:4px;">SMUNet</td>
      <td style="border:1px solid #000; padding:4px;">75.0</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">67.9</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">69.6</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">84.2</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">76.7</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">70.6</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">84.6</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">77.1</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">85.2</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">85.2</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">85.4</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">85.6</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">86.0</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">77.2</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">86.0</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">79.7</span>
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #000; padding:4px;">MST-KDNet</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">77.2</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">72.9</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">73.5</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">84.7</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">79.8</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">75.1</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">85.7</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">79.3</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">85.8</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">86.4</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">86.5</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">86.1</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">86.9</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">80.0</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">86.8</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">81.8</span>
      </td>
    </tr>
    
    <!-- TC 行数据 -->
    <tr>
      <td style="border:1px solid #000; padding:4px;" rowspan="7">TC</td>
      <td style="border:1px solid #000; padding:4px;">RA-HVED</td>
      <td style="border:1px solid #000; padding:4px;">26.5</td>
      <td style="border:1px solid #000; padding:4px;">54.2</td>
      <td style="border:1px solid #000; padding:4px;">9.4</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">41.1</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">61.3</td>
      <td style="border:1px solid #000; padding:4px;">54.8</td>
      <td style="border:1px solid #000; padding:4px;">41.9</td>
      <td style="border:1px solid #000; padding:4px;">29.2</td>
      <td style="border:1px solid #000; padding:4px;">40.5</td>
      <td style="border:1px solid #000; padding:4px;">61.9</td>
      <td style="border:1px solid #000; padding:4px;">62.5</td>
      <td style="border:1px solid #000; padding:4px;">43.2</td>
      <td style="border:1px solid #000; padding:4px;">64.0</td>
      <td style="border:1px solid #000; padding:4px;">61.9</td>
      <td style="border:1px solid #000; padding:4px;">65.0</td>
      <td style="border:1px solid #000; padding:4px;">47.8</td>
    </tr>
    <tr>
      <td style="border:1px solid #000; padding:4px;">RMBTS</td>
      <td style="border:1px solid #000; padding:4px;">10.9</td>
      <td style="border:1px solid #000; padding:4px;">36.5</td>
      <td style="border:1px solid #000; padding:4px;">12.6</td>
      <td style="border:1px solid #000; padding:4px;">11.2</td>
      <td style="border:1px solid #000; padding:4px;">40.4</td>
      <td style="border:1px solid #000; padding:4px;">37.6</td>
      <td style="border:1px solid #000; padding:4px;">16.8</td>
      <td style="border:1px solid #000; padding:4px;">15.2</td>
      <td style="border:1px solid #000; padding:4px;">14.5</td>
      <td style="border:1px solid #000; padding:4px;">38.9</td>
      <td style="border:1px solid #000; padding:4px;">40.1</td>
      <td style="border:1px solid #000; padding:4px;">17.4</td>
      <td style="border:1px solid #000; padding:4px;">40.4</td>
      <td style="border:1px solid #000; padding:4px;">40.9</td>
      <td style="border:1px solid #000; padding:4px;">40.6</td>
      <td style="border:1px solid #000; padding:4px;">27.6</td>
    </tr>
    <tr>
      <td style="border:1px solid #000; padding:4px;">mmformer</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">47.2</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">52.3</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">44.4</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">33.1</td>
      <td style="border:1px solid #000; padding:4px;">62.6</td>
      <td style="border:1px solid #000; padding:4px;">60.6</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">49.6</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">51.1</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">49.6</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">60.6</td>
      <td style="border:1px solid #000; padding:4px;">64.3</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">52.6</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">65.5</td>
      <td style="border:1px solid #000; padding:4px;">65.3</td>
      <td style="border:1px solid #000; padding:4px;">67.0</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">55.1</span>
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #000; padding:4px;">M2FTrans</td>
      <td style="border:1px solid #000; padding:4px;">46.6</td>
      <td style="border:1px solid #000; padding:4px;">53.3</td>
      <td style="border:1px solid #000; padding:4px;">43.3</td>
      <td style="border:1px solid #000; padding:4px;">33.8</td>
      <td style="border:1px solid #000; padding:4px;">60.0</td>
      <td style="border:1px solid #000; padding:4px;">57.7</td>
      <td style="border:1px solid #000; padding:4px;">46.7</td>
      <td style="border:1px solid #000; padding:4px;">48.5</td>
      <td style="border:1px solid #000; padding:4px;">48.3</td>
      <td style="border:1px solid #000; padding:4px;">57.8</td>
      <td style="border:1px solid #000; padding:4px;">60.0</td>
      <td style="border:1px solid #000; padding:4px;">49.6</td>
      <td style="border:1px solid #000; padding:4px;">61.5</td>
      <td style="border:1px solid #000; padding:4px;">60.8</td>
      <td style="border:1px solid #000; padding:4px;">62.0</td>
      <td style="border:1px solid #000; padding:4px;">52.7</td>
    </tr>
    <tr>
      <td style="border:1px solid #000; padding:4px;">ACN</td>
      <td style="border:1px solid #000; padding:4px;">21.2</td>
      <td style="border:1px solid #000; padding:4px;">54.2</td>
      <td style="border:1px solid #000; padding:4px;">19.5</td>
      <td style="border:1px solid #000; padding:4px;">22.5</td>
      <td style="border:1px solid #000; padding:4px;">58.8</td>
      <td style="border:1px solid #000; padding:4px;">57.9</td>
      <td style="border:1px solid #000; padding:4px;">26.1</td>
      <td style="border:1px solid #000; padding:4px;">23.2</td>
      <td style="border:1px solid #000; padding:4px;">26.7</td>
      <td style="border:1px solid #000; padding:4px;">60.0</td>
      <td style="border:1px solid #000; padding:4px;">63.8</td>
      <td style="border:1px solid #000; padding:4px;">28.3</td>
      <td style="border:1px solid #000; padding:4px;">62.6</td>
      <td style="border:1px solid #000; padding:4px;">62.7</td>
      <td style="border:1px solid #000; padding:4px;">64.1</td>
      <td style="border:1px solid #000; padding:4px;">43.4</td>
    </tr>
    <tr>
      <td style="border:1px solid #000; padding:4px;">SMUNet</td>
      <td style="border:1px solid #000; padding:4px;">29.3</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">64.1</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">28.2</td>
      <td style="border:1px solid #000; padding:4px;">28.8</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">67.3</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">67.1</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">32.6</td>
      <td style="border:1px solid #000; padding:4px;">31.5</td>
      <td style="border:1px solid #000; padding:4px;">32.5</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">66.9</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">70.4</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">33.7</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">69.4</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">69.1</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">69.8</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">50.7</td>
    </tr>
    <tr>
      <td style="border:1px solid #000; padding:4px;">MST-KDNet</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">47.3</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">68.3</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">44.5</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">33.9</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">70.3</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">71.3</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">50.1</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">41.5</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">50.2</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">72.0</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">74.1</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">53.6</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">72.5</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">72.6</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">73.1</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">59.5</span>
      </td>
    </tr>

    <!-- ET 行数据 -->
    <tr>
      <td style="border:1px solid #000; padding:4px;" rowspan="7">ET</td>
      <td style="border:1px solid #000; padding:4px;">RA-HVED</td>
      <td style="border:1px solid #000; padding:4px;">35.8</td>
      <td style="border:1px solid #000; padding:4px;">37.8</td>
      <td style="border:1px solid #000; padding:4px;">9.24</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">39.8</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">42.3</td>
      <td style="border:1px solid #000; padding:4px;">36.6</td>
      <td style="border:1px solid #000; padding:4px;">42.6</td>
      <td style="border:1px solid #000; padding:4px;">43.8</td>
      <td style="border:1px solid #000; padding:4px;">44.4</td>
      <td style="border:1px solid #000; padding:4px;">44.1</td>
      <td style="border:1px solid #000; padding:4px;">43.9</td>
      <td style="border:1px solid #000; padding:4px;">48.4</td>
      <td style="border:1px solid #000; padding:4px;">46.8</td>
      <td style="border:1px solid #000; padding:4px;">40.7</td>
      <td style="border:1px solid #000; padding:4px;">45.9</td>
      <td style="border:1px solid #000; padding:4px;">40.1</td>
    </tr>
    <tr>
      <td style="border:1px solid #000; padding:4px;">RMBTS</td>
      <td style="border:1px solid #000; padding:4px;">7.9</td>
      <td style="border:1px solid #000; padding:4px;">37.8</td>
      <td style="border:1px solid #000; padding:4px;">10.0</td>
      <td style="border:1px solid #000; padding:4px;">8.2</td>
      <td style="border:1px solid #000; padding:4px;">41.9</td>
      <td style="border:1px solid #000; padding:4px;">40.1</td>
      <td style="border:1px solid #000; padding:4px;">13.1</td>
      <td style="border:1px solid #000; padding:4px;">11.8</td>
      <td style="border:1px solid #000; padding:4px;">10.8</td>
      <td style="border:1px solid #000; padding:4px;">40.6</td>
      <td style="border:1px solid #000; padding:4px;">43.5</td>
      <td style="border:1px solid #000; padding:4px;">14.0</td>
      <td style="border:1px solid #000; padding:4px;">42.3</td>
      <td style="border:1px solid #000; padding:4px;">44.1</td>
      <td style="border:1px solid #000; padding:4px;">55.2</td>
      <td style="border:1px solid #000; padding:4px;">28.1</td>
    </tr>
    <tr>
      <td style="border:1px solid #000; padding:4px;">mmformer</td>
      <td style="border:1px solid #000; padding:4px;">44.9</td>
      <td style="border:1px solid #000; padding:4px;">50.5</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">42.3</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">31.4</td>
      <td style="border:1px solid #000; padding:4px;">61.3</td>
      <td style="border:1px solid #000; padding:4px;">59.0</td>
      <td style="border:1px solid #000; padding:4px;">45.3</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">49.4</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">46.6</td>
      <td style="border:1px solid #000; padding:4px;">59.3</td>
      <td style="border:1px solid #000; padding:4px;">63.0</td>
      <td style="border:1px solid #000; padding:4px;">49.6</td>
      <td style="border:1px solid #000; padding:4px;">63.6</td>
      <td style="border:1px solid #000; padding:4px;">64.2</td>
      <td style="border:1px solid #000; padding:4px;">65.7</td>
      <td style="border:1px solid #000; padding:4px;">53.1</td>
    </tr>
    <tr>
      <td style="border:1px solid #000; padding:4px;">M2FTrans</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">47.1</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">54.2</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">44.6</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">34.0</td>
      <td style="border:1px solid #000; padding:4px;">62.6</td>
      <td style="border:1px solid #000; padding:4px;">60.0</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">47.5</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">49.4</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">49.3</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">60.2</td>
      <td style="border:1px solid #000; padding:4px;">62.7</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">50.4</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">64.5</td>
      <td style="border:1px solid #000; padding:4px;">63.4</td>
      <td style="border:1px solid #000; padding:4px;">65.0</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">54.3</span>
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #000; padding:4px;">ACN</td>
      <td style="border:1px solid #000; padding:4px;">18.0</td>
      <td style="border:1px solid #000; padding:4px;">55.2</td>
      <td style="border:1px solid #000; padding:4px;">16.9</td>
      <td style="border:1px solid #000; padding:4px;">19.6</td>
      <td style="border:1px solid #000; padding:4px;">59.8</td>
      <td style="border:1px solid #000; padding:4px;">59.6</td>
      <td style="border:1px solid #000; padding:4px;">22.2</td>
      <td style="border:1px solid #000; padding:4px;">19.2</td>
      <td style="border:1px solid #000; padding:4px;">22.4</td>
      <td style="border:1px solid #000; padding:4px;">60.8</td>
      <td style="border:1px solid #000; padding:4px;">65.1</td>
      <td style="border:1px solid #000; padding:4px;">23.9</td>
      <td style="border:1px solid #000; padding:4px;">64.0</td>
      <td style="border:1px solid #000; padding:4px;">64.3</td>
      <td style="border:1px solid #000; padding:4px;">65.9</td>
      <td style="border:1px solid #000; padding:4px;">42.5</td>
    </tr>
    <tr>
      <td style="border:1px solid #000; padding:4px;">SMUNet</td>
      <td style="border:1px solid #000; padding:4px;">25.5</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">64.8</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">25.0</td>
      <td style="border:1px solid #000; padding:4px;">25.1</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">67.9</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">68.1</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">28.6</td>
      <td style="border:1px solid #000; padding:4px;">27.6</td>
      <td style="border:1px solid #000; padding:4px;">28.6</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">67.9</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">70.6</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">29.7</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">69.8</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">70.1</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:blue;">70.8</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">49.3</td>
    </tr>
    <tr>
      <td style="border:1px solid #000; padding:4px;">MST-KDNet</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">48.3</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">68.6</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">32.0</td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">40.6</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">70.0</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">72.3</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">48.5</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">50.1</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">51.1</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">72.4</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">74.9</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">52.5</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">72.8</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">73.1</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">73.9</span>
      </td>
      <td style="border:1px solid #000; padding:4px;">
        <span style="color:red;">59.8</span>
      </td>
    </tr>
  </tbody>
</table>


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
