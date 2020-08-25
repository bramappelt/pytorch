# Pytorch

Investigate runtimes using a cpu and a gpu as training devices.

The code is adapted from [the pytorch tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) and answers the following question:

_Exercise: Try increasing the width of your network (argument 2 of the first nn.Conv2d, and argument 1 of the second nn.Conv2d – they need to be the same number), see what kind of speedup you get._

![Training times for both devices and the model's accuracy for increasing neural net widths](./figure1.png).

To replicate the experiment clone the repo and run `benchmark.py`, Python 3.6 is was used.

## Hardware

- CPU: Intel i9-8950HK CPU @ 2.90GHz
- GPU: Nvidia Geforce GTX 1050 Ti

Note: The CUDA driver should be installed on your GPU.
