export LD_LIBRARY_PATH=/home/mengchen/gpu_benchmarks/nccl/build/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NCCL_BUFFSIZE=1048576
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,P2P,GRAPH
#export NCCL_GRAPH_FILE=./graph.xml
#export NCCL_ALGO=RING #Tree

#export NCCL_P2P_DISABLE=1

#CORE_FREQ="1.0 3.8"
CORE_FREQ="3.8"
PERF_COMD="all_reduce alltoall"
PERF_COMD="all_reduce"
#PERF_COMD="sendrecv"
#PERF_COMD="reduce_scatter"
#PERF_COMD="all_gather"
#PERF_COMD="alltoall"
#INST="SYS-PIX"
#INST="PIX-PIX-PIX-PIX"
INST="SYS-SYS-SYS-SYS"
#INST="SYS"
#INST="LOC"
#INST="PIX"
#INST="NVL"

# set cpu/freq mode
cpupower frequency-info -g performance
cpupower idle-set -E

# set uncore freq
wrmsr -a 0x620 0x1919

# enable DDIO,default value is c0000 (rdmsr -a 0xc8b)
#wrmsr -a 0xc8b 0xffff

START=1048000
END=2048000000
FACTOR=2
NG=2

for freq in ${CORE_FREQ}
do
cpupower frequency-set -u ${freq}g -d ${freq}g
#     turbostat -i1 -c "0-5,48-53" -n 10 &
for cmd in ${PERF_COMD}
do

instid=0

OLD_IFS=$IFS
IFS="-"
for p2p in $INST;
do
IFS=$OLD_IFS
gid=`expr $instid + 4`
#gid=`expr $instid + 1`
#CUDA_VISIBLE_DEVICES="`expr $instid`, `expr $instid + 1`, `expr $instid + 2`, `expr $instid + 3`" \
#CUDA_VISIBLE_DEVICES="$instid, $gid" \
#NCCL_TOPO_DUMP_FILE=./${cmd}_${INST}_${instid}_${p2p}_topo.xml NCCL_GRAPH_DUMP_FILE=./${cmd}_${INST}_${instid}_${p2p}_graph.xml \
CUDA_VISIBLE_DEVICES="$instid, $gid" \
NCCL_P2P_LEVEL=$p2p ../nccl-tests/build/${cmd}_perf -b $START -e $END -f $FACTOR -g $NG 2>&1 | tee ${cmd}_${INST}_${instid}_${p2p}_${freq}.log &

#instid=`expr $instid + 4`
#instid=`expr $instid + 2`
instid=`expr $instid + 1`

done
done
done

# back to default
cpupower frequency-set -u 3.8g -d 0.8g > /dev/null 2>&1
#wrmsr -a 0xc8b 0xc0000
wrmsr -a 0x620 0x818
