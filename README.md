## Spectro-Spatio-Temporal EEG Representation Learning for Imagined Speech Recognition
<p align="center"><img width="90%" src="files/dipnet.png" /></p>

This repository provides a TensorFlow implementation of the following paper:
> **VIGNet: A Deep Convolutional Neural Network for EEG-based Driver Vigilance Estimation**<br>
> [Wonjun Ko](https://scholar.google.com/citations?user=Fvzg1_sAAAAJ&hl=ko&oi=ao)<sup>1</sup>, [Eunjin Jeon](https://scholar.google.com/citations?user=U_hg5B0AAAAJ&hl=ko)<sup>1</sup>, [Heung-Il Suk](https://scholar.google.co.kr/citations?user=dl_oZLwAAAAJ&hl=ko)<sup>1, 2</sup><br/>
> (<sup>1</sup>Department of Brain and Cognitive Engineering, Korea University) <br/>
> (<sup>2</sup>Department of Artificial Intelligence, Korea University) <br/>
> [Official version]: TBA
> [Presented in the 6th Asian Conference on Pattern Recognition (ACPR)](http://brain.korea.ac.kr/acpr/)
> 
> **Abstract:** *In brain–computer interfaces, imagined speech is one of the most promising paradigms due to its intuitiveness and direct communication. However, it is challenging to decode an imagined speech EEG, because of its complicated underlying cognitive processes, resulting in complex spectro-spatio-temporal patterns. In this work, we propose a novel convolutional neural network structure for representing such complex patterns and identifying an intended imagined speech. The proposed network exploits two feature extraction flows for learning richer class-discriminative information. Specifically, our proposed network is composed of a spatial filtering path and a temporal structure learning path running in parallel, then integrates their output features for decision-making. We demonstrated the validity of our proposed method on a publicly available dataset by achieving state-of-the-art performance. Furthermore, we analyzed our network to show that our method learns neurophysiologically plausible patterns.*

## Dependencies
* [Python 3.7+](https://www.continuum.io/downloads)
* [TensorFlow 2.0.0+](https://www.tensorflow.org/)

## Downloading datasets
To download 2020 International BCI Competition Track 3 dataset
* https://osf.io/pq7vb/
