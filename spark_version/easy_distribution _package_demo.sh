#!/bin/bash
####This script in on the master  machine;The worker1 host is hadoop103 and worker2 host is hadoop104(line23~25).
####example: easy_distribution_package_demo.sh test.txt
pcount=$#
if ((pcount==0)); then
echo no args;
exit;
fi


p1=$1
fname=`basename $p1`
echo fname=$fname


pdir=`cd -P $(dirname $p1); pwd`
echo pdir=$pdir


user=`whoami`


for((host=103; host<105; host++)); do
        echo ------------------- hadoop$host --------------
        rsync -av $pdir/$fname $user@hadoop$host:$pdir 
done
