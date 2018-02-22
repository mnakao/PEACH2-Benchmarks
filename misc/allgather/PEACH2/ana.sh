FILE=$1
awk '{print $2}' $FILE | head -n  32 | tail -n 16  > 0.txt
awk '{print $2}' $FILE | head -n  64 | tail -n 16 > 1.txt
awk '{print $2}' $FILE | head -n  96 | tail -n 16 > 2.txt
awk '{print $2}' $FILE | head -n 128 | tail -n 16 > 3.txt
awk '{print $2}' $FILE | head -n 160 | tail -n 16 > 4.txt
awk '{print $2}' $FILE | head -n 192 | tail -n 16 > 5.txt
awk '{print $2}' $FILE | head -n 224 | tail -n 16 > 6.txt
awk '{print $2}' $FILE | head -n 256 | tail -n 16 > 7.txt
awk '{print $2}' $FILE | head -n 288 | tail -n 16 > 8.txt
awk '{print $2}' $FILE | head -n 320 | tail -n 16 > 9.txt
paste *.txt
rm -f *.txt

