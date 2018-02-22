#PBS -S /bin/bash
#PBS -A TCAGEN
#PBS -N overQPI
#PBS -q tcaq-q1
#PBS -l select=2:ncpus=1
#PBS -l walltime=00:03:00
module purge
module load cuda/7.5.18 intel/16.0.4 mvapich2-gdr/2.2_intel_cuda-7.5
#OPT="MV2_USE_GPUDIRECT_RECEIVE_LIMIT=8192 CUDA_VISIBLE_DEVICES=0 MV2_ENABLE_AFFINITY=0 MV2_USE_CUDA=1 numactl --cpunodebind=0 --localalloc"
OPT="MV2_USE_GPUDIRECT_RECEIVE_LIMIT=2048 MV2_USE_GPUDIRECT_GDRCOPY=0 CUDA_VISIBLE_DEVICES=0 MV2_ENABLE_AFFINITY=0 MV2_USE_CUDA=1 numactl --cpunodebind=0 --localalloc"

cd $PBS_O_WORKDIR
export LD_LIBRARY_PATH=../../fujita/lib:$LD_LIBRARY_PATH
for i in $(seq 1 10)
do
mpirun_rsh -hostfile $PBS_NODEFILE -np 2 $OPT ./ping.out
done
echo $SECONDS
