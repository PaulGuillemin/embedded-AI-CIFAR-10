import os
import subprocess

############### Configurations ########################
model = "cnn_quan"

clipping_value = 0.0
randbet = 0
lr = 0.01

seeds = ['5555', '758', '3666', '4258', '6213']
attack_id=0

for seed in seeds:
    print('************************')
    print('****** BFA ', attack_id+1, '/5 ', '******')
    print('************************')
    attack_id += 1
    cmd = [
        "python3", "main.py",
        "--arch", model,
        "--bfa",
        "--manualSeed", seed,
        "--learning_rate", str(lr),
        "--clipping_coeff", str(clipping_value)
        ]

    if randbet == 1:
        cmd.extend(["--randbet"])

    # Run the command using subprocess
    subprocess.run(cmd)
