FILE=$1
grep -v ERROR $FILE | awk '{print $6}' | head -n 22  | tail -n 20 > 1.txt
grep -v ERROR $FILE | awk '{print $6}' | head -n 44  | tail -n 20 > 2.txt
grep -v ERROR $FILE | awk '{print $6}' | head -n 66  | tail -n 20 > 3.txt

paste *.txt
rm -f *.txt
