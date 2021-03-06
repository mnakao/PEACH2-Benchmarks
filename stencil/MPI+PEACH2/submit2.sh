#PBS -S /bin/bash
#PBS -A TCAGEN
#PBS -N Stencil
#PBS -q tcaq-q1
#PBS -l select=1:ncpus=1:host=tcag-0015+1:ncpus=1:host=tcag-0016+1:ncpus=1:host=tcag-0014+1:ncpus=1:host=tcag-0013+1:ncpus=1:host=tcag-0012
#PBS -l walltime=00:01:00
module purge
module load cuda/7.5.18 intel/16.0.4 mvapich2-gdr/2.2_intel_cuda-7.5
OPT="MV2_USE_GPUDIRECT_RECEIVE_LIMIT=8192 MV2_USE_GPUDIRECT_GDRCOPY=0 CUDA_VISIBLE_DEVICES=0 MV2_ENABLE_AFFINITY=0 MV2_USE_CUDA=1 numactl --cpunodebind=0 --localalloc"

cd $PBS_O_WORKDIR
export LD_LIBRARY_PATH=../../misc/fujita/lib:$LD_LIBRARY_PATH
for i in $(seq 1 1)
do
mpirun_rsh -hostfile $PBS_NODEFILE -np 5 $OPT ./mpi-peach2-stencil.out
done
