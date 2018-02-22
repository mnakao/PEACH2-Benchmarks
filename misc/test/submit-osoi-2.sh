#PBS -S /bin/bash
#PBS -A TCAGEN
#PBS -N osoi-2
#PBS -q tcaq-q1
#PBS -l select=1:ncpus=1:host=tcag-0004+1:ncpus=1:host=tcag-0016
#PBS -l walltime=00:03:00
module purge
module load cuda/7.5.18 intel/16.0.4 mvapich2-gdr/2.2_intel_cuda-7.5
OPT="numactl --cpunodebind=0 --localalloc"

cd $PBS_O_WORKDIR
for i in $(seq 1 3)
do
mpirun_rsh -hostfile $PBS_NODEFILE -np 2 $OPT ./ping.out
done

