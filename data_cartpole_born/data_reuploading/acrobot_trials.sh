#!/bin/bash

N=5  # Maximum number of parallel jobs
M=20  # Total number of jobs
job_pool=()  # Array to hold current jobs

# Function to remove a specific element from an array
array_remove() {
    local value="$1"
    for i in "${!job_pool[@]}"; do
        if [ "${job_pool[i]}" == "$value" ]; then
            unset "job_pool[i]"
            break
        fi
    done
}


# Launch jobs
for j in $(seq 1 $M); do
    while [ "${#job_pool[@]}" -ge "$N" ]; do
        # Check each job in the pool to see if it has finished
        for job in "${job_pool[@]}"; do
            if ! kill -0 "$job" 2>/dev/null; then
                array_remove "$job"
            fi
        done
        # Sleep for a brief moment to prevent CPU overload
        sleep 1
    done

    # Run the Python script as a background job and get its PID
    python qpg_acrobot.py --n_layers 5 --episodes 500 --init normal_0_1 --batch_size 10 --kl_divergence 1 --meyer_wallach_entanglement 1 --data_reuploading True --measurement n-local --born_policy global --environment Acrobot-v1 --comparator_policy product-approx&
    job_pool+=("$!")

done

# Launch jobs
for j in $(seq 1 $M); do
    while [ "${#job_pool[@]}" -ge "$N" ]; do
        # Check each job in the pool to see if it has finished
        for job in "${job_pool[@]}"; do
            if ! kill -0 "$job" 2>/dev/null; then
                array_remove "$job"
            fi
        done
        # Sleep for a brief moment to prevent CPU overload
        sleep 1
    done

    # Run the Python script as a background job and get its PID
    python qpg_acrobot.py --n_layers 5 --episodes 500 --init normal_0_1 --batch_size 10 --kl_divergence 0 --meyer_wallach_entanglement 1 --data_reuploading True --measurement n-local --born_policy modulo --environment Acrobot-v1 --comparator_policy global&
    job_pool+=("$!")

done

# Wait for remaining jobs to finish
for job in "${job_pool[@]}"; do
    wait "$job"
done