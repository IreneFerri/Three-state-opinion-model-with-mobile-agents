#!/bin/bash

# ---- d = 0.250 ------------------------------

export LC_NUMERIC="en_US.UTF-8"   # IMPORTANT for printf variables

chmod +x mak* # Execute permission

cp neigh_vctt.py neigh_v0.01.py 
v_array=($(seq -f "%f" 0.01 0.02 0.06 ))
indexes=( $(seq 0 2 ))
for i in ${indexes[@]}
do
    echo $i
    echo ${v_array[i+1]}
    m=${v_array[i]} 
    j=${v_array[i+1]}
    echo $j
    printf "%.2f\n" ${j} 
    printf -v next "%.2f" ${j}  
    echo $next
    printf -v current "%.2f" ${m}   
    echo $current

    sed "s/vmin = 0.01/vmin = ${current}/g" neigh_vctt.py > neigh_v${current}_temp.py
    sed "s/vmax = 0.02/vmax = ${next}/g" neigh_v${current}_temp.py > neigh_v${current}.py 
    rm neigh_v${current}_temp.py
    python3 neigh_v${current}.py
done

