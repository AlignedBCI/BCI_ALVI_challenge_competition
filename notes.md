250 epochs: 0.901
2500 epochs: 0.75


# Connecting to OPENMING
ssh om-login


CPU: srun -p fiete -n 4 -t 600 --mem 20G --pty bash
GPU: srun -p fiete -n 4 -t 600 --mem 60G --gres=gpu:4 --constraint=any-gpu --pty bash

Then use `hostname` and replace that in the SSH config. 