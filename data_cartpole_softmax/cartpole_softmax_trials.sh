#!/bin/bash

N=5
for j in $(seq 1 20)
do
	((i=i%N)); ((i++==0)) && wait
	python qnpg_reinforce_cartpole_SOFTMAX.py --init glorot --policy Q --ng 1 --n_layers 4 --episodes 1000 --entanglement all2all --batch_size 4 &
done

#N=5
#for j in $(seq 1 20)
#do
	#((i=i%N)); ((i++==0)) && wait
	#python qnpg_reinforce_cartpole_SOFTMAX.py --init glorot --policy Q --ng 1 --n_layers 4 --episodes 1000 --entanglement all2all --batch_size 4 &
#done

