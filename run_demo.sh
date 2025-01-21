export LD_LIBRARY_PATH=/home/mengchen/gpu_benchmarks/nccl/build/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=SYS #NVL
export NCCL_BUFFSIZE=1048576
#export NCCL_DEBUG_SUBSYS=INIT,P2P,GRAPH

# set cpu/freq mode
cpupower frequency-info -g performance
cpupower idle-set -E

cpupower frequency-set -u 3.8g -d 3.8g
# set uncore freq
wrmsr -a 0x620 0x1919

# enable DDIO,default value is c0000 (rdmsr -a 0xc8b)
wrmsr -a 0xc8b 0xffff

NPROC=8
PMODE=1

if [ -z $PMODE ]; then
    WITERS="0 1"
fi

NG_PER_PROC=`expr 8 \/ $NPROC`

mpirun -n $NPROC \
        --allow-run-as-root \
        ./demo $NG_PER_PROC $WITERS


# back to default
cpupower frequency-set -u 3.8g -d 0.8g > /dev/null 2>&1
wrmsr -a 0xc8b 0xc0000
wrmsr -a 0x620 0x818
