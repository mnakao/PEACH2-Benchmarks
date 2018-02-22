FILE=$1
awk '{print $4}' $FILE | head -n 17 > 0.txt
awk '{print $4}' $FILE | head -n 34 | tail -n 17 > 1.txt
awk '{print $4}' $FILE | head -n 51 | tail -n 17 > 2.txt
awk '{print $4}' $FILE | head -n 68 | tail -n 17 > 3.txt
awk '{print $4}' $FILE | head -n 85 | tail -n 17 > 4.txt
awk '{print $4}' $FILE | head -n 102 | tail -n 17 > 5.txt
awk '{print $4}' $FILE | head -n 119 | tail -n 17 > 6.txt
awk '{print $4}' $FILE | head -n 136 | tail -n 17 > 7.txt
awk '{print $4}' $FILE | head -n 153 | tail -n 17 > 8.txt
awk '{print $4}' $FILE | head -n 170 | tail -n 17 > 9.txt
paste *.txt
rm -f *.txt

