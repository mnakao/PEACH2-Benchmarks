FILE=$1
awk '{print $4}' $FILE | head -n 20 > 0.txt
awk '{print $4}' $FILE | head -n 40 | tail -n 20 > 1.txt
awk '{print $4}' $FILE | head -n 60 | tail -n 20 > 2.txt
awk '{print $4}' $FILE | head -n 80 | tail -n 20 > 3.txt
awk '{print $4}' $FILE | head -n 100 | tail -n 20 > 4.txt
awk '{print $4}' $FILE | head -n 120 | tail -n 20 > 5.txt
awk '{print $4}' $FILE | head -n 140 | tail -n 20 > 6.txt
awk '{print $4}' $FILE | head -n 160 | tail -n 20 > 7.txt
awk '{print $4}' $FILE | head -n 180 | tail -n 20 > 8.txt
awk '{print $4}' $FILE | head -n 200 | tail -n 20 > 9.txt
paste *.txt
rm -f *.txt

