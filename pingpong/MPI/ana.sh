FILE=$1
grep -v ERROR $FILE | awk '{print $6}' | head -n 20 > 1.txt
grep -v ERROR $FILE | awk '{print $6}' | head -n 40  | tail -n 20 > 2.txt
grep -v ERROR $FILE | awk '{print $6}' | head -n 60  | tail -n 20 > 3.txt
grep -v ERROR $FILE | awk '{print $6}' | head -n 80  | tail -n 20 > 4.txt
grep -v ERROR $FILE | awk '{print $6}' | head -n 100 | tail -n 20 > 5.txt
grep -v ERROR $FILE | awk '{print $6}' | head -n 120 | tail -n 20 > 6.txt
grep -v ERROR $FILE | awk '{print $6}' | head -n 140 | tail -n 20 > 7.txt
grep -v ERROR $FILE | awk '{print $6}' | head -n 180 | tail -n 20 > 8.txt
grep -v ERROR $FILE | awk '{print $6}' | head -n 200 | tail -n 20 > 9.txt
grep -v ERROR $FILE | awk '{print $6}' | head -n 220 | tail -n 21 > 0.txt

paste *.txt
rm -f *.txt
