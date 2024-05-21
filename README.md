# MAP
The source code of our works on federated learning:
* TKDE 2024 paper: MAP: Model Aggregation and Personalization in Federated Learning With Incomplete Classes. IEEE Transactions on Knowledge and Data Engineering. DOI: 10.1109/TKDE.2024.3390041.


# Content
* Personal Homepage
* Basic Introduction
* Code Files
* Running Tips
* Citation

## Personal Homepage
  * [Homepage](https://www.lamda.nju.edu.cn/lixc/)

## Basic Introduction
  * \[**Considered Scene**\] We focus on a special kind of Non-I.I.D. scene where clients own incomplete classes, i.e., each client can only access a partial set of the whole class set.
  * \[**Aggregation and Personalization**\] The server aims to aggregate a complete classification model that could generalize to all classes, while the clients are inclined to improve the performance of distinguishing their observed classes.
  * \[**Proposed Method**\] For better model aggregation, we point out that the standard softmax will encounter several problems caused by missing classes and propose “restricted softmax” as an alternative. For better model personalization, we point out that the hard-won personalized models are not well exploited and propose “inherited private model” to store the personalization experience. Our proposed algorithm named MAP could simultaneously achieve the aggregation and personalization goals in FL.

## Compared FL Algorithms
We implement several popular FL algorithms with local regularization (e.g., FedProx, FedDyn), better optimization (e.g., FedNova, FedOpt), control variates (e.g., Scaffold), contrastive learning (e.g., MOON), etc. FedRS and FedPHP are our previous proposed methods. Some methods are for better aggregation (e.g., FedOpt, Scaffold, FedRS), while some are for better personalization (e.g., pFedMe, PerFedAvg, FedPHP). FedROD simultaneously considers the aggregation and personalization.

The compred FL algorithms are implemented based on the our published FL repository, i.e., [FedRepo](https://github.com/lxcnju/FedRepo). Differently, we record both the aggregation and personalization performances for each algorithm. These algorithms could be found in the directory of `algorithms/'.

## Environment Dependencies
The code files are written in Python, and the utilized deep learning tool is PyTorch.
  * `python`: 3.7.3
  * `numpy`: 1.21.5
  * `torch`: 1.9.0
  * `torchvision`: 0.10.0
  * `pillow`: 8.3.1

## Datasets
We provide several datasets including (downloading link code be found in my [Homepage](https://www.lamda.nju.edu.cn/lixc/)):
  * FaMnist
  * CIFAR-10
  * CIFAR-100

These codes are encapsulated into the `datasets/feddata.py`.


## Running Tips
  * `python train_demo.py`: run algorithms as a demo which sets the number of local epochs as 1, the number of communication round as 2, and the client participation ratio as 0.02;
  * `python train.py`: run algorithms that use the settings in our paper; the hyperparameters are searched in 3 groups and the best results are reported.

FL algorithms and hyper-parameters could be set in these files.


## Citation
  * Xin-Chun Li, Shaoming Song, Yinchuan Li, Bingshuai Li, Yunfeng Shao, Yang Yang, De-Chuan Zhan. MAP: Model Aggregation and Personalization in Federated Learning With Incomplete Classes. In: IEEE Transactions on Knowledge and Data Engineering. DOI: 10.1109/TKDE.2024.3390041.
  * \[[BibTex](https://dblp.org/pid/246/2947.html)\]
