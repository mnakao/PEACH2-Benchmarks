#PBS -S /bin/bash
#PBS -A TCAGEN
#PBS -N no-overQPI
#PBS -q tcaq-q1
#PBS -l select=2:ncpus=1
#PBS -l walltime=00:03:00
module purge
module load cuda/7.5.18 intel/16.0.4 mvapich2-gdr/2.2_intel_cuda-7.5
#OPT="MV2_GPUDIRECT_LIMIT=524288 CUDA_VISIBLE_DEVICES=2 MV2_ENABLE_AFFINITY=0 MV2_USE_CUDA=1 numactl --cpunodebind=1 --localalloc"
OPT="MV2_GPUDIRECT_LIMIT=262144 CUDA_VISIBLE_DEVICES=2 MV2_ENABLE_AFFINITY=0 MV2_USE_CUDA=1 numactl --cpunodebind=1 --localalloc"

cd $PBS_O_WORKDIR
export LD_LIBRARY_PATH=../../fujita/lib:$LD_LIBRARY_PATH
for i in $(seq 1 10)
do
mpirun_rsh -hostfile $PBS_NODEFILE -np 2 $OPT ./ping.out
done
echo $SECONDS