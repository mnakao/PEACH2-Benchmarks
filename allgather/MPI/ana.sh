FILE=$1
awk '{print $4}' $FILE | head -n 16 > 0.txt
awk '{print $4}' $FILE | head -n 32 | tail -n 16 > 1.txt
awk '{print $4}' $FILE | head -n 48 | tail -n 16 > 2.txt
awk '{print $4}' $FILE | head -n 64 | tail -n 16 > 3.txt
awk '{print $4}' $FILE | head -n 80 | tail -n 16 > 4.txt
awk '{print $4}' $FILE | head -n 96 | tail -n 16 > 5.txt
awk '{print $4}' $FILE | head -n 112 | tail -n 16 > 6.txt
awk '{print $4}' $FILE | head -n 128 | tail -n 16 > 7.txt
awk '{print $4}' $FILE | head -n 144 | tail -n 16 > 8.txt
awk '{print $4}' $FILE | head -n 160 | tail -n 16 > 9.txt
paste *.txt
rm -f *.txt

