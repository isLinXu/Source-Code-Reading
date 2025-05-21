#!/bin/bash

# $1: save_dir
# $2: exp_name
# $3: previously saved step
# $4: current saved step
# Example use: ./sync_and_upload.sh /home/victor/experiments vllama_debug opt_step-40 opt_step-50

if [[ -n "$4" ]]; then
    if [[ -n "$3" ]]; then
        s5cmd sync "$1/$2/$3/" "s3://m4-exps/$2/$3/" && rm -rf "$1/$2/$3"
    fi
    s5cmd cp "$1/$2/$4/" "s3://m4-exps/$2/$4/"
fi
