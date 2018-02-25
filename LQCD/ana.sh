FILE=$1
grep "performance of solver" $FILE | grep GFlops | awk '{print $5}'
grep AVE $FILE | awk '{print $3"\t"$5"\t"$7"\t"$9"\t"$11"\t"$13"\t"$15"\t"$17}'