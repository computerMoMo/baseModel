#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python MF.py --test 0 --epoch 50 --loss_type square_loss --test_file_path ../Data/user_item_test_0.0.txt