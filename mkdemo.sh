nvcc -I/usr/local/cuda-12.5/targets/x86_64-linux/include/ \
    -L/usr/local/cuda-12.5/targets/x86_64-linux/lib -lcudart \
    -I/home/mengchen/gpu_benchmarks/nccl/build/include \
    -L/home/mengchen/gpu_benchmarks/nccl/build/lib -lnccl \
    -I/home/mengchen/miniconda3/envs/py310/include/ \
    -L/home/mengchen/miniconda3/envs/py310/lib/ -lmpi \
    -O3 \
    demo.cc -o demo
