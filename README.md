#  Pruning Convolutional Neural Networks in the Darknet framework

**Author:** [Pedro Vinícius A. B. Venâncio](https://www.linkedin.com/in/pedbrgs/)<sup>1,2</sup> <br />

> <sup>1</sup> Graduate Program in Electrical Engineering ([PPGEE](https://www.ppgee.ufmg.br/indexi.php)/[UFMG](https://ufmg.br/international-visitors))<br />
> <sup>2</sup> Gaia, solutions on demand ([GAIA](https://www.gaiasd.com/))<br />

***

### About

This repository contains the source codes of some techniques for filter pruning in convolutional neural networks implemented in the [Darknet framework](https://github.com/AlexeyAB/darknet/). Pruning a trained network with an appropriate technique can slightly decrease its performance (or even improve it in some cases), in addition to making the network lighter and faster to run on mobile devices and machines without a Graphics Processing Unit (GPU).
The pruning techniques implemented can be classified into three categories:

1. Criteria-based pruning techniques: L0-Norm, L1-Norm, L2-Norm, L-Inf Norm and Random.
2. Projection-based pruning techniques: PLS(Single)+VIP, PLS(Multi)+VIP, CCA(Multi)+CV and PLS(Multi)+LC.
3. Cluster-based pruning techniques: HAC+PCC.

Note: If you do not want to reuse the filters that remained in the pruned architecture for fine-tuning, just use the From-Scratch mode.

***

### Tutorial

After compiling [Darknet](https://github.com/AlexeyAB/darknet/) on your machine, creating your .cfg and .data files and training your network, follow these steps:

1. Download this repository within your `darknet/` folder:

```
git clone https://github.com/pedbrgs/Pruning-Darknet/ darknet/pruning/
```

2. After that, move all files and folders from `pruning/` to root directory `darknet/`. You can remove the `pruning/` folder, which is empty now.

3. Move the `.cfg`, `.data`, `.names`, `.weights`, `train.txt` and `valid.txt` files into the darknet repository. Make sure that you have already trained the network for the desired task and that the paths in `.data` are correct.

Note: If you arrived here in this repository without ever training a network in the [Darknet framework](https://github.com/AlexeyAB/darknet/), I recommend that you learn how to use it first. You need to compile Darknet, define your deep network, define your dataset and train your model to proceed.

4. Run pruning with fine-tuning:

* Example for a criterion-based method:

```python
python prune.py --cfg yolo.cfg --data yolo.data --names yolo.names --weights yolo.weights --network YOLOv4 --img-size 416 --technique L1 --pruning-rate 0.60 --pruning-iter 2 --lr 0.005 --tuning --tuning-iter 30000
```

* Example for a projection-based method:

```python
python prune.py --cfg yolo.cfg --data yolo.data --names yolo.names --weights yolo.weights --network YOLOv4 --img-size 416 --technique PLS-VIP-Multi --pruning-rate 0.60 --pruning-iter 2 --lr 0.005 --pool-type max --n-components 2 --num-classes 3 --perc-samples 0.1 --tuning --tuning-iter 30000
```

* Example for a cluster-based method:

```python
python prune.py --cfg yolo.cfg --data yolo.data --names yolo.names --weights yolo.weights --network YOLOv4 --img-size 416 --technique HAC --pruning-rate 0.60 --pruning-iter 2 --lr 0.005 --measure Pearson --tuning --tuning-iter 30000`
```

***

### Citation info

Please cite my paper in your publications if you use this source code.

- <p align="justify"><b>Pedro Vinícius Almeida Borges de Venâncio</b>, Adriano Chaves Lisboa, Adriano Vilela Barbosa: <a href="https://link.springer.com/article/10.1007/s00521-022-07467-z"> An automatic fire detection system based on deep convolutional neural networks for low-power, resource-constrained devices. </a> In: Neural Computing and Applications, 2022.</p>

***

### Useful links about Darknet

* [Darknet Manual](https://github.com/AlexeyAB/darknet/wiki)
* [Compile on Windows](https://github.com/AlexeyAB/darknet/#how-to-compile-on-windows-using-cmake)
* [Compile on Linux/macOS](https://github.com/AlexeyAB/darknet/#how-to-compile-on-linux-using-make)
* [Requirements (and how to install dependecies)](https://github.com/AlexeyAB/darknet/#requirements)

***

### Contact
Please send any bug reports, questions or suggestions directly in the repository.
