start_idx=3000
num_files=3
num_cores=600
street=4

total=`expr $num_files \* $num_cores \+ $start_idx`

for idx in `seq $start_idx $num_files $total`
do
 echo $idx
 sbatch ./run_cfr.sh $street $idx
done
