#!/usr/bin/env bash
id_arrays=("0.2" "0.4" "0.6" "0.8" "1.0")
for idx in ${id_arrays[@]}
do
    echo $idx
    test_file_name="../Data/user_item_test_"$idx".txt"
    CUDA_VISIBLE_DEVICES=0 python MF.py --test 1 --model_epoch 50 --test_file_path $test_file_name
    break
done