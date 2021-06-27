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

<a name="data_RGB" />

#### Datasets recorded with commercial RGB camera

| Dataset    | Year | #Live/Spoof | #Sub. |  Setup | Attack Types |
| --------   | -----    | -----  |  -----  | ----- |------------------------|
| [NUAA](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Searching_Central_Difference_Convolutional_Networks_for_Face_Anti-Spoofing_CVPR_2020_paper.pdf)   | 2010 | 5105/7509(I) | 15 |  N/R | Print(flat, wrapped)|
| [YALE Recaptured](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Searching_Central_Difference_Convolutional_Networks_for_Face_Anti-Spoofing_CVPR_2020_paper.pdf)   | 2011 | 640/1920(I) | 10 |  50cm-distance from 3 LCD minitors | Print(flat) |
| [CASIA-MFSD](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Searching_Central_Difference_Convolutional_Networks_for_Face_Anti-Spoofing_CVPR_2020_paper.pdf)   | 2012 | 150/450(V) | 50 |  7 scenarios and 3 image quality | Print(flat, wrapped, cut), Replay(tablet)|
| [REPLAY-ATTACK](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Searching_Central_Difference_Convolutional_Networks_for_Face_Anti-Spoofing_CVPR_2020_paper.pdf)   | 2012 | 200/1000(V) | 50 |  Lighting and holding | Print(flat), Replay(tablet, phone) |
| [Kose and Dugelay](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Searching_Central_Difference_Convolutional_Networks_for_Face_Anti-Spoofing_CVPR_2020_paper.pdf)   | 2013 | 200/198(I) | 20 |  N/R | Mask(hard resin) |
| [MSU-MFSD](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Searching_Central_Difference_Convolutional_Networks_for_Face_Anti-Spoofing_CVPR_2020_paper.pdf)   | 2014 | 70/210(V) | 35 |  Indoor scenario; 2 types of cameras | Print(flat), Replay(tablet, phone) |
| [UVAD](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Searching_Central_Difference_Convolutional_Networks_for_Face_Anti-Spoofing_CVPR_2020_paper.pdf)   | 2015 | 808/16268(V) | 404 | Different lighting, background and places in two sections | Replay(monitor) |
| [REPLAY-Mobile](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Searching_Central_Difference_Convolutional_Networks_for_Face_Anti-Spoofing_CVPR_2020_paper.pdf)   | 2016 | 390/640(V) | 40 |  5 lighting conditions | Print(flat), Replay(monitor) |
| [HKBU-MARs V2](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Searching_Central_Difference_Convolutional_Networks_for_Face_Anti-Spoofing_CVPR_2020_paper.pdf)   | 2016 | 504/504(V) | 12 | 7 cameras from stationary and mobile devices and 6 lighting settings | Mask(hard resin) from Thatsmyface and REAL-f |
| [MSU USSA](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Searching_Central_Difference_Convolutional_Networks_for_Face_Anti-Spoofing_CVPR_2020_paper.pdf)   | 2016 | 1140/9120(I) | 1140 |  Uncontrolled; 2 types of cameras | Print(flat), Replay(laptop, tablet, phone)|
| [SMAD](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Searching_Central_Difference_Convolutional_Networks_for_Face_Anti-Spoofing_CVPR_2020_paper.pdf)   | 2017 | 65/65(V) | - |  Color images from online resources | Mask(silicone) |
| [OULU-NPU](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Searching_Central_Difference_Convolutional_Networks_for_Face_Anti-Spoofing_CVPR_2020_paper.pdf)   | 2017 | 720/2880(V) | 55 |  Lighting & background in 3 sections | Print(flat), Replay(phone) |
| [Rose-Youtu](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Searching_Central_Difference_Convolutional_Networks_for_Face_Anti-Spoofing_CVPR_2020_paper.pdf)   | 2018 | 500/2850(V) | 20 | 5 front-facing phone camera; 5 different illumination conditions | Print(flat), Replay(monitor, laptop),Mask(paper, crop-paper)|
| [SiW](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Searching_Central_Difference_Convolutional_Networks_for_Face_Anti-Spoofing_CVPR_2020_paper.pdf)   | 2018 | 1320/3300(V) | 165 |  4 sessions with variations of distance, pose, illumination and expression | Print(flat, wrapped), Replay(phone, tablet, monitor)|
| [WFFD](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Searching_Central_Difference_Convolutional_Networks_for_Face_Anti-Spoofing_CVPR_2020_paper.pdf)   | 2019 | 2300/2300(I) 140/145(V) | 745 |  Collected online; super-realistic; removed low-quality faces | Waxworks(wax)|



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
    
