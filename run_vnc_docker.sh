#!/bin/bash

docker run \
       --rm \
       -p 5902:5900 \
       -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
       -v /dev/shm:/dev/shm \
       -v /home/duke/sawyer_comp/mjkey/mjkey.txt:/root/.mujoco/mjkey.txt \
       -v /home/duke/sawyer_comp/code/:/root/code/ \
       -v /home/duke/sawyer_comp/robosuite/:/root/robosuite/ \
       -v /home/duke/sawyer_comp/stable-baselines/:/root/stable-baselines/ \
       --env DISPLAY=$DISPLAY \
       --env CUDA_VISIBLE_DEVICES=3 \
       --runtime nvidia \
       crl_vnc_sawyer 
