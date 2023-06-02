# cuDNN correctness tests

Verify if the operator's calculation results are correct on other gpu products compatible with NVIDIA CUDA/cuDNN.

## Building

```
mkdir build
cd build
cmake ..
make
```

Note: The most recent cuDNN distribution will be obtained automatically by installing PyTorch into the build directory.

## Testing

Run all correctness tests at once:

```
cd build
make
ctest
```

Run all correctness tests at once, with CUDA/cuDNN stack intercepted by GPUOcelot:

```
cd build
cmake -DENABLE_OCELOT=/home/marcusmae/gpuocelot/gpuocelot/ocelot/build/libgpuocelot.so ..
make
ctest
```

## Generating the ground truth data

We generate input/weight/bias/outout data with TensorFlow API. For example, we use `tf.nn.maxpool()`, and then compare against it the result of a call to `cudnnMaxpoolForward`:

1. Generate input/output data:

```
python tf-maxpooling.py
```

2. Paste new input/output data into `float32.h`:

```
dtype input[IN_SIZE] = {...}
dtype output[OUT_SIZE] = {...}
```

3. Re-run the test to verify `cudnnMaxpoolForward`.

