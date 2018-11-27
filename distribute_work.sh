start_idx=0
num_files=3
num_cores=10
street=4

for idx in `seq $start_idx $num_files $num_cores`
do
 ./run_cfr.sh $idx $street
done
