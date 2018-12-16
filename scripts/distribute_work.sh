start_idx=3000
num_files=3
num_cores=600
street=4
approximate='root_nodes'

total=`expr $num_files \* $num_cores \+ $start_idx`

for idx in `seq $start_idx $num_files $total`
do
 echo $idx
 sbatch -o ../out/$idx ./run_cfr.sh $street $idx $approximate
done
