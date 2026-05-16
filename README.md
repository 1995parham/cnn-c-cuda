# CNN implementation in C + CUDA :dog:

[![build](https://github.com/1995parham/cnn-c-cuda/actions/workflows/build.yml/badge.svg)](https://github.com/1995parham/cnn-c-cuda/actions/workflows/build.yml)

##### based on [cnn-c-cuda](https://github.com/lucasromanosantos/cnn-c-cuda)

A small Convolutional Neural Network in C with optional CUDA kernels, trained on the
[MNIST dataset](http://yann.lecun.com/exdb/mnist/) for handwritten digit recognition.

## Architecture

```text
input              28 x 28 x 1
  └── conv  (5x5, 8 filters, stride 1)   ->  24 x 24 x 8
  └── relu                               ->  24 x 24 x 8
  └── pool  (2x2, stride 2, max)         ->  12 x 12 x 8
  └── fc                                 ->  10
```

Only the forward pass is GPU-accelerated; training (backward + weight update)
runs on the CPU.

## Build

The MNIST IDX files are expected under `data/` (`train-images.idx3-ubyte`,
`train-labels.idx1-ubyte`, `t10k-images.idx3-ubyte`, `t10k-labels.idx1-ubyte`).

CPU-only build (gcc):

```sh
make cpu
```

GPU build (requires the CUDA toolkit / `nvcc`):

```sh
make
```

Both produce a `main` binary that loads MNIST, trains on up to 5000 examples,
and runs inference over the test set.

## CI

GitHub Actions builds both targets on every push and pull request to `main`
(see [`.github/workflows/build.yml`](.github/workflows/build.yml)). The CUDA
job installs the toolkit and only verifies that the kernels compile — there is
no GPU on the hosted runners.

## Credits

Heavily inspired by [can1357/simple_cnn](https://github.com/can1357/simple_cnn).
MNIST loader by [Nuri Park](https://github.com/projectgalateia/mnist).
