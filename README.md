<div align="center"> Official repository of <br> &nbsp; </div>

# <div align="center"> SyPer </div>

<div align="center">
  <p>
    <img width="50%" src="https://github.com/jankolf/SyPer/blob/main/images/syper_logo.png?raw=true">
  </p>

# <div align="center"> Synthetic periocular data for quantized light-weight recognition in the NIR and visible domains </div>

<div align="center", padding="30px">
  <span style="display:block; height: 20em;">&emsp;</span>
  <p><b>Jan Niklas Kolf</b><sup> 1,2</sup> &emsp; <b>Jurek Elliesen</b><sup> 1</sup> &emsp; <b>Fadi Boutros</b><sup> 1</sup> &emsp; <b>Hugo Proença</b><sup> 3</sup> &emsp; <b>Naser Damer</b><sup> 1,2</sup>  </p>
  <p><sup>1 </sup>Fraunhofer IGD &emsp; <sup>2 </sup>Technische Universität Darmstadt  &emsp; <sup>3 </sup>University of Beira Interior, IT: Instituto de Telecomunicações</p>
  <span style="display:block; height: 50em;">&emsp;</span>
  <p><b>Image and Vision Computing</b> <br> Volume 135, July 2023</p>
  <span style="display:block; height: 50em;">&emsp;</span>
</div>

<div align="center">
        <a href="https://sharedata.jankolf.de/syperpaper"><img src="https://github.com/jankolf/assets/blob/main/badges/download-paper-c1.svg?raw=true"></a>
        &emsp;
        <a href="https://sharedata.jankolf.de/syper"><img src="https://github.com/jankolf/assets/blob/main/badges/download-data-c1.svg?raw=true"></a>
</div>

</div>

## <div align="center">SyPer 🧐</div>

### Abstract
Deep-learning based periocular recognition systems typically use overparameterized deep neural networks associated with high computational costs and memory requirements. This is especially problematic for mobile and embedded devices in shared resource environments.
To perform model quantization for lightweight periocular recognition in a privacy-aware manner, we propose and release SyPer, a synthetic dataset and generation model of periocular images. 
To enable this, we propose to perform the knowledge transfer in the quantization process on the embedding level and thus not identity-labeled data. This does not only allow the use of synthetic data for quantization, but it also successfully allows to perform the quantization on different domains to additionally boost the performance in new domains.
In a variety of experiments on a diverse set of model backbones, we demonstrate the ability to build compact and accurate models through an embedding-level knowledge transfer using synthetic data. We also demonstrate very successfully the use of embedding-level knowledge transfer for near-infrared quantized models towards accurate and efficient periocular recognition on near-infrared images.

### Graphical Abstract
<div align="center">
  <p>
    <img width="80%" src="https://github.com/jankolf/SyPer/blob/main/images/graphical_abstract.png?raw=true">
  </p>
</div>


## <div align="center"> Citation ✒ </div>
Please cite the article with the following bibtex entry:
```
@article{KOLF2023104692,
title = {SyPer: Synthetic periocular data for quantized light-weight recognition in the NIR and visible domains},
journal = {Image and Vision Computing},
pages = {104692},
year = {2023},
issn = {0262-8856},
doi = {https://doi.org/10.1016/j.imavis.2023.104692},
url = {https://www.sciencedirect.com/science/article/pii/S0262885623000665},
author = {Jan Niklas Kolf and Jurek Elliesen and Fadi Boutros and Hugo Proença and Naser Damer},
keywords = {Deep learning, Quantization, Synthetic data, Biometrics, Periocular},
abstract = {Deep-learning based periocular recognition systems typically use overparameterized deep neural networks associated with high computational costs and memory requirements. This is especially problematic for mobile and embedded devices in shared resource environments. To perform model quantization for lightweight periocular recognition in a privacy-aware manner, we propose and release SyPer, a synthetic dataset and generation model of periocular images. To enable this, we propose to perform the knowledge transfer in the quantization process on the embedding level and thus not identity-labeled data. This does not only allow the use of synthetic data for quantization, but it also successfully allows to perform the quantization on different domains to additionally boost the performance in new domains. In a variety of experiments on a diverse set of model backbones, we demonstrate the ability to build compact and accurate models through an embedding-level knowledge transfer using synthetic data. We also demonstrate very successfully the use of embedding-level knowledge transfer for near-infrared quantized models towards accurate and efficient periocular recognition on near-infrared images. The SyPer dataset, together with the evaluation protocol, the training code, and model checkpoints are made publicly available at https://github.com/jankolf/SyPer.}
}
```

## <div align="center"> License 📄 </div>
```
This project is licensed under the terms of the Attribution-NonCommercial-ShareAlike 4.0 
International (CC BY-NC-SA 4.0) license. 
Copyright (c) 2021 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt
```
