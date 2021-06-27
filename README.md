# 👏 Survey of Deep Face Anti-spoofing 🔥

This is the official repository of **Deep Learning for Face Anti-Spoofing: A Survey**, a comprehensive survey 
of recent progress in deep learning methods for face anti-spoofing (FAS) as well as the datasets and protocols.


## Introduction
We present a comprehensive review of recent deep learning methods for face anti-spoofing (mostly from 2018 to 2021). It covers hybrid (handcrafted+deep), pure deep learning, and generalized learning based methods for monocular RGB face anti-spoofing. It also includes multi-modal learning based methods as well as specialized sensor based FAS. It also presents detailed comparision among publicly available datasets, together with several classical evaluation protocols.

🔔 We will update this page frequently~ :tada::tada::tada:

---
## Contents

- [Datasets](#data)
  - [Using commercial RGB camera](#data_RGB)
  - [With multiple modalities or specialized sensors](#data_Multimodal)
- [Deep FAS methods with commercial RGB camera](#methods_RGB)
  - [Hybrid (handcrafted + deep)](#hybrid)
  - [Pure binary cross-entropy supervision](#binary)
  - [Pixel-wise auxiliary supervision](#auxiliary)
  - [Generative model with pixel-wise supervision](#generative)
  - [Domain adaptation](#DA)
  - [Domain generalization](#DG)
  - [Zero/Few-shot learning](#zero-shot)
  - [Anomaly detection](#oneclass)
- [Deep FAS methods with advanced sensor](#methods_advanced)
  - [Learning upon specialized sensor](#sensor)
  - [Multi-modal learning](#multimodal)
  
---


<a name="data" />

### 1️⃣ Datasets

- temp

<a name="data_RGB" />

#### Datasets recorded with commercial RGB camera

| Dataset    | Year | #Live/Spoof | #Sub. |  Setup | Attack Types
| --------   | -----    | -----  |  -----  | ----- |------|------------------------|
| NUAA  [PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Searching_Central_Difference_Convolutional_Networks_for_Face_Anti-Spoofing_CVPR_2020_paper.pdf)   | 2010 | 5105/7509(I) | 15 |  N/R | Print(flat, wrapped)|

<a name="data_Multimodal" />

#### Datasets with multiple modalities or specialized sensors

|Method    | xx| xx  | xx |  xx |  xx| Paper
| --------   | -----    | -----  |  -----  | ----- |------|------------------------|
|xx     | xx | xx% | xx%|  xx% |xx |xx [PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Searching_Central_Difference_Convolutional_Networks_for_Face_Anti-Spoofing_CVPR_2020_paper.pdf)|

---
<a name="methods_RGB" />

### 2️⃣ Deep FAS methods with commercial RGB camera

- temp
- 
<a name="hybrid" />

#### Hybrid (handcrafted + deep)


<a name="binary" />

#### Pure binary cross-entropy supervision


<a name="auxiliary" />

#### Pixel-wise auxiliary supervision

<a name="generative" />

#### Generative model with pixel-wise supervision

<a name="DA" />

#### Domain adaptation

<a name="DG" />

#### Domain generalization

<a name="zero-shot" />

#### Zero/Few-shot learning

<a name="oneclass" />

#### Anomaly detection



---
<a name="methods_advanced" />

### 3️⃣ Deep FAS methods with advanced sensor


<a name="sensor" />

#### Learning upon specialized sensor

<a name="multimodal" />

#### Multi-modal learning

---

### Citation
If you find our work useful in your research, please consider citing:

    @article{yu2021deep,
      title={Deep Learning for Face Anti-Spoofing: A Survey},
      author={Yu, Zitong and Qin, Yunxiao and Li, Xiaobai and Zhao, Chenxu and Lei, Zhen and Zhao, Guoying},
      journal={arXiv},
      year={2021}
    }
    
