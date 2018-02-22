FILE=$1
grep -v ERROR $FILE | awk '{print $6}' | head -n 22  | tail -n 20 > 1.txt
grep -v ERROR $FILE | awk '{print $6}' | head -n 44  | tail -n 20 > 2.txt
grep -v ERROR $FILE | awk '{print $6}' | head -n 66  | tail -n 20 > 3.txt
grep -v ERROR $FILE | awk '{print $6}' | head -n 88  | tail -n 20 > 4.txt
grep -v ERROR $FILE | awk '{print $6}' | head -n 110 | tail -n 20 > 5.txt
grep -v ERROR $FILE | awk '{print $6}' | head -n 132 | tail -n 20 > 6.txt
grep -v ERROR $FILE | awk '{print $6}' | head -n 154 | tail -n 20 > 7.txt
grep -v ERROR $FILE | awk '{print $6}' | head -n 176 | tail -n 20 > 8.txt
grep -v ERROR $FILE | awk '{print $6}' | head -n 198 | tail -n 20 > 9.txt
grep -v ERROR $FILE | awk '{print $6}' | head -n 220 | tail -n 20 > 0.txt

paste *.txt
rm -f *.txt
