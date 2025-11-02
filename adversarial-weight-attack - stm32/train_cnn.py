import os
import subprocess



clipping_value = 0.0
randbet = 0
lr = 0.01




############### Neural network ############################
cmd = [
    "python3", "main.py",
    "--arch", "cnn_quan",
    "--learning_rate", str(lr),
    "--clipping_coeff", str(clipping_value)
]

if randbet == 1:
   cmd.extend(["--randbet"])


print(f"Executing command: {' '.join(cmd)}")

# Execute the command
subprocess.run(cmd)
