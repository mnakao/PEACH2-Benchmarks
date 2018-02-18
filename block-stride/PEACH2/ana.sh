FILE=$1
awk '{print $6}' $FILE | head -n 9 > 1.txt
awk '{print $6}' $FILE | head -n 18 | tail -n 9 > 2.txt
awk '{print $6}' $FILE | head -n 27 | tail -n 9 > 3.txt
awk '{print $6}' $FILE | head -n 36 | tail -n 9 > 4.txt
awk '{print $6}' $FILE | head -n 45 | tail -n 9 > 5.txt
awk '{print $6}' $FILE | head -n 54 | tail -n 9 > 6.txt
awk '{print $6}' $FILE | head -n 63 | tail -n 9 > 7.txt
awk '{print $6}' $FILE | head -n 72 | tail -n 9 > 8.txt
awk '{print $6}' $FILE | head -n 81 | tail -n 9 > 9.txt
awk '{print $6}' $FILE | head -n 90 | tail -n 9 > 10.txt

paste *.txt
rm -f *.txt
