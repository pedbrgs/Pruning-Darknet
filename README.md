##  Pruning techniques of Convolutional Neural Networks implemented in the Darknet framework

**Author:** [Pedro Vinícius A. B. Venâncio](https://www.linkedin.com/in/pedbrgs/)<sup>1,2</sup> <br />

> <sup>1</sup> Graduate Program in Electrical Engineering ([PPGEE](https://www.ppgee.ufmg.br/indexi.php)/[UFMG](https://ufmg.br/international-visitors))<br />
> <sup>2</sup> Gaia, solutions on demand ([GAIA](https://www.gaiasd.com/))<br />

***

### About

This repository contains the source codes of some techniques for filter pruning in convolutional neural networks implemented in the [Darknet framework](https://github.com/AlexeyAB/darknet/). Pruning a trained network with an appropriate technique can slightly decrease its performance (or even improve it in some cases), in addition to making the network lighter and faster to run on mobile devices and machines without a Graphics Processing Unit (GPU) .
The pruning techniques implemented can be classified into two categories:

1. Criteria-based pruning techniques: L0-Norm, L1-Norm, L2-Norm, L-Inf Norm and Random.
2. Projection-based pruning techniques: PLS-VIP-Single, PLS-VIP-Multi, CCA-CV-Multi and PLS-LC-Multi.

Note: If you arrived here in this repository without ever training a network in the Darknet framework, I recommend that you learn how to use it first.
You need to compile Darknet, define your deep network, define your dataset and train your model to proceed.
***

### Tutorial

After cloning this repository, move the `.cfg`, `.data`, `.names` and `.weights` files into it. 
If you chose a projection-based pruning technique, be sure to get the variables for projection before with `scripts/get_variables.py`.

For a criterion-based method, an example is to run the following command:

`python prune.py --cfg yolo.cfg --data yolo.data --names yolo.names --weights yolo.weights --network YOLOv4 --img-size 416 --technique L1 --pruning-rate 0.60`

For a projection-based method, an example is to run the following command:

`python prune.py --cfg yolo.cfg --data yolo.data --names yolo.names --weights yolo.weights --network YOLOv4 --img-size 416 --technique PLS-VIP-Multi --pruning-rate 0.60 --n-components 2 --variables variables.npy`



***

### Useful links about Darknet

* [Darknet Manual](https://github.com/AlexeyAB/darknet/wiki)
* [Compile on Windows](https://github.com/AlexeyAB/darknet/#how-to-compile-on-windows-using-cmake)
* [Compile on Linux/macOS](https://github.com/AlexeyAB/darknet/#how-to-compile-on-linux-using-make)
* [Requirements (and how to install dependecies)](https://github.com/AlexeyAB/darknet/#requirements)

***

### Citation info

If you're using these codes in any way, please let them know your source:

```
@Misc{Venancio2021-Pruning,
title = {Pruning techniques of convolutional neural networks implemented in the Darknet framework},
author = {Pedro Vinicius A. B. Venancio},
howPublished = {\url{https://git.io/JmjNB}},
year = {2021}}
```

***

### Contact
Please send any bug reports, questions or suggestions directly in the repository.
