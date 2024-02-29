#!/bin/bash


N=5
for j in $(seq 1 10)
do
	((i=i%N)); ((i++==0)) && wait
	python qpg_reinforce_cartpole_partitions.py --n_layers 4 --episodes 500 --init normal_0_1 --batch_size 10 --kl_divergence 0 --meyer_wallach_entanglement 0 --data_reuploading True --measurement n-local --born_policy global --environment CartPole-v0 --comparator_policy optimal&
done

for j in $(seq 1 10)
do
	((i=i%N)); ((i++==0)) && wait
	python qpg_reinforce_cartpole_partitions.py --n_layers 4 --episodes 500 --init normal_0_1 --batch_size 10 --kl_divergence 1 --meyer_wallach_entanglement 0 --data_reuploading True --measurement n-local --born_policy product-approx --environment CartPole-v0 --comparator_policy optimal&
done

for j in $(seq 1 10)
do
	((i=i%N)); ((i++==0)) && wait
	python qpg_reinforce_cartpole_partitions.py --n_layers 4 --episodes 500 --init normal_0_1 --batch_size 10 --kl_divergence 1 --meyer_wallach_entanglement 0 --data_reuploading True --measurement n-local --born_policy global --environment CartPole-v0 --comparator_policy optimal&
done
