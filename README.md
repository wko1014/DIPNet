## Spectro-Spatio-Temporal EEG Representation Learning for Imagined Speech Recognition
<p align="center"><img width="90%" src="files/dipnet.png" /></p>

This repository provides a TensorFlow implementation of the following paper:
> **Spectro-Spatio-Temporal EEG Representation Learning for Imagined Speech Recognition**<br>
> [Wonjun Ko](https://scholar.google.com/citations?user=Fvzg1_sAAAAJ&hl=ko&oi=ao)<sup>1</sup>, [Eunjin Jeon](https://scholar.google.com/citations?user=U_hg5B0AAAAJ&hl=ko)<sup>1</sup>, and [Heung-Il Suk](https://scholar.google.co.kr/citations?user=dl_oZLwAAAAJ&hl=ko)<sup>1, 2</sup><br/>
> (<sup>1</sup>Department of Brain and Cognitive Engineering, Korea University) <br/>
> (<sup>2</sup>Department of Artificial Intelligence, Korea University) <br/>
> Official Version: TBA<br/>
> [Presented in the 6th Asian Conference on Pattern Recognition (ACPR)](http://brain.korea.ac.kr/acpr/)
> 
> **Abstract:** *In brain–computer interfaces, imagined speech is one of the most promising paradigms due to its intuitiveness and direct communication. However, it is challenging to decode an imagined speech EEG, because of its complicated underlying cognitive processes, resulting in complex spectro-spatio-temporal patterns. In this work, we propose a novel convolutional neural network structure for representing such complex patterns and identifying an intended imagined speech. The proposed network exploits two feature extraction flows for learning richer class-discriminative information. Specifically, our proposed network is composed of a spatial filtering path and a temporal structure learning path running in parallel, then integrates their output features for decision-making. We demonstrated the validity of our proposed method on a publicly available dataset by achieving state-of-the-art performance. Furthermore, we analyzed our network to show that our method learns neurophysiologically plausible patterns.*

and a submitted paper:
> **DIPNet: Dual Information Pathways Network for Unarticulated Speech Recognition**<br>
> [Wonjun Ko](https://scholar.google.com/citations?user=Fvzg1_sAAAAJ&hl=ko&oi=ao)<sup>1</sup>, [Eunjin Jeon](https://scholar.google.com/citations?user=U_hg5B0AAAAJ&hl=ko)<sup>1</sup>, and [Heung-Il Suk](https://scholar.google.co.kr/citations?user=dl_oZLwAAAAJ&hl=ko)<sup>1, 2</sup><br/>
> (<sup>1</sup>Department of Brain and Cognitive Engineering, Korea University) <br/>
> (<sup>2</sup>Department of Artificial Intelligence, Korea University) <br/>
> Official Version: Submitted<br/>
> 
> **Abstract:** *Imagined and inner speech have attracted increasing attention in brain--computer interface research circles, owing to its ability to enable direct and intuitive communication between users and external devices. However, as these imagined/inner speech electroencephalograms (EEGs) are created through complex and internal cognitive process, recognizing informative patterns within these unarticulated EEGs is a challenge. In this work, we devise a novel deep convolutional neural network architecture to extract spatial filtering information and temporal dynamics of input imagined/inner speech EEGs. Specifically, we consider two different information representation paths and employ these paths in a unified architecture complementarily, thereby effectively learning complex EEG features. To demonstrate the validity of the proposed network architecture, we conduct subject-dependent and independent classification experiments with comparable state-of-the-art methods on publicly available imagined speech inner speech EEG datasets. Furthermore, we observe that each path used in our architecture learns different information from the input signal through extensive analyses.*

## Dependencies
* [Python 3.7+](https://www.continuum.io/downloads)
* [TensorFlow 2.0.0+](https://www.tensorflow.org/)

## Downloading datasets
To download 2020 International BCI Competition Track 3 dataset
* https://osf.io/pq7vb/

## Usage
`network.py` contains the proposed deep learning architectures, `utils.py` contains functions used for experimental procedures, `experiment.py` contains the main experimental functions, and `main.py' is the main function.

## Citation
If you find this work useful for your research, please cite our paper: TBA
```
TBA
```

## Acknowledgements
This work was supported by Institute for Information & Communications Technology Promotion (IITP) grant funded by the Korea government under Grant 2017-0-00451 (Development of BCI based Brain and Cognitive Computing Technology for Recognizing User’s Intentions using Deep Learning) and Grant 2019-0-00079 (Department of Artificial Intelligence, Korea University).
