#!/bin/bash

N=5
for j in $(seq 1 20)
do
	((i=i%N)); ((i++==0)) && wait
	python qnpg_reinforce_acrobot_born.py --init glorot --policy Q --ng 0 --n_layers 5 --episodes 1000 --entanglement all2all --batch_size 4 --meyer_wallach 0 &
done


#for j in $(seq 1 20)
#do
	#((i=i%N)); ((i++==0)) && wait
	#python qnpg_reinforce_cartpole_born.py --init glorot --policy Q --ng 0 --n_layers 5 --episodes 1000 --entanglement all2all --batch_size 4 --meyer_wallach 1 &
#done

