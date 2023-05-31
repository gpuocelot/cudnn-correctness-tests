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

# how to use
for example, we try to generate data(e.g, input/weight/bias/outout) with tensorflow api, such as tf.nn.maxpool(), and then using this data to verify the correctness of cudnnMaxpoolForward api.

```
# generate input/output data 
$ python tf-maxpooling.py
# modify float32.h to replace input/output data.
dtype input[IN_SIZE] ={...}
dtype output[OUT_SIZE] = {...}
# verify cudnnMaxpoolForward api.
$ make
$ ./test
```
