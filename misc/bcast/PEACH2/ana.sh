FILE=$1
awk '{print $2}' $FILE | head -n  36 | tail -n 20 > 1.txt
awk '{print $2}' $FILE | head -n  72 | tail -n 20 > 2.txt
awk '{print $2}' $FILE | head -n 108 | tail -n 20 > 3.txt
awk '{print $2}' $FILE | head -n 144 | tail -n 20 > 4.txt
awk '{print $2}' $FILE | head -n 180 | tail -n 20 > 5.txt
awk '{print $2}' $FILE | head -n 216 | tail -n 20 > 6.txt
awk '{print $2}' $FILE | head -n 252 | tail -n 20 > 7.txt
awk '{print $2}' $FILE | head -n 288 | tail -n 20 > 8.txt
awk '{print $2}' $FILE | head -n 324 | tail -n 20 > 9.txt
awk '{print $2}' $FILE | head -n 360 | tail -n 20 > 0.txt
paste *.txt
rm -f *.txt

