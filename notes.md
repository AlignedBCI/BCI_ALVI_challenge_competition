0.118


# Connecting to OPENMING
ssh om-login


CPU: srun -p fiete -n 24 -t 600 --mem 20G --pty bash
GPU: srun -p fiete -n 24 -t 600 --mem 20G --gres=gpu:1 --constraint=any-gpu --pty bash

Then use `hostname` and replace that in the SSH config. 