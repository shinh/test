eval `date '+y=%Y;m=$((%-m+(%e>13)))'`
for i in `seq $y 2013`;do
for j in `seq $m 12`;do
if [ `cal $j $i|awk '$6==13&&$_=$6'` ];then
c=$((c+1))
printf $i-%02d-13\\n $j
fi
done
m=1
done
echo $c
