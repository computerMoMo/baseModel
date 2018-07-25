#!/usr/bin/env bash
test_array=("0.2" "0.4" "0.6" "0.8" "1.0")
for idx in ${test_array[@]}
do
    echo $idx
    python ItemPop.py $idx
done