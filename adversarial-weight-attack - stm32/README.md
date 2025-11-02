# :dart: Tutorial :dart: Adversarial Weight Attack

## Authors

Kevin HECTOR, Pierre-Alain MOELLIC

## Objective

This tutorial aims to provide a simple example on the Bit-Flip Attack [Rakin et al., ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Rakin_Bit-Flip_Attack_Crushing_Neural_Network_With_Progressive_Bit_Search_ICCV_2019_paper.pdf).

We provide Python scripts (PyTorch) to experiment some basic attack & defense principles related to the BFA:

- Two datasets: **CIFAR-10** and **MIT-BIH** (1d ECG traces)
- For **CIFAR-10** the target model is a **RESNET20**. For **MIT-BIH** the target model is a small **CNN** (Conv1d). 
- Train models with and without training-based defenses (NB: models are quantified on 8 bits)
- Display the distribution of the weights and the gradients (of the loss wrt weights)
- Apply the BFA 
- Display the impact of the bit-flips on the accuracy 

## Protections 

In this tutorial, we use 3 protections against the BFA: 

-  Learning rate variation [Hector et al., IOLTS 2022](https://arxiv.org/abs/2209.14243)
-  Parameter clipping
-  RandBET (robust training) [Stutz et al., IEEE TPAMI 2021](https://arxiv.org/abs/2104.08323). 

## Available pretrained models (./save)

For RESNET20 on CIFAR10, we provide 4 pretrained models (in the ./save/ directory) demonstrating the impact of these protections. Each model can be selected using 3 variables in all scripts: the Learning rate **'lr'**, the value of clipping **'clipping_value'** and a binary flag for **'randbet'**.

- Nominal model: 'lr=0.1', 'clipping_value=0.0' and 'randbet=0'
- model trained with a lower learning rate: 'lr=0.01', 'clipping_value=0.0' and 'randbet=0'
- model protected by weights clipping: 'lr=0.1', 'clipping_value=0.1' and 'randbet=0'
- model protected by weights clipping + RandBET: 'lr=0.1', 'clipping_value=0.1' and 'randbet=1'

## Custom the TinyVGG model (Cifar-10) [2025 - ISMIN EI23]

For Cifar-10, we provide a TinyVGG model () 

You have to replace this TinyVGG model with your optimized model (see "Usage" section below)

To do so, you need to modify the architecture in **models/quan_vgg.py**

## Custom the 1D-CNN model (MIT-BIH) [old 2024 - ISMIN EI22]

For MIT-BIH, you need to train the CNN model

The available architecture is not optimal (conv1D[8 filters size=3] + MaxPool + Flatten + Dense[10] + Dense[5]) 

You need to improve it by adapting the architecture in **quan_mit_bih.py**


## Scripts

- training script is **train_{your_model}.py** (i.e., *train_cnn.py*, *train_resnet20.py*n and *train_tinyvgg.py*)
- attack script (BFA) is **bfa_{your_model}.py**
- NB: both scripts use the main.py script that encapsulates all the code.
- To plot the performance drop as a function of the number of bit-flips used **printing_tools.py**.

**Cf. Usage below for an example (CNN on MIT-BIH)**

## Installation

With anaconda: 

    conda create --name bfa_tuto
    conda activate bfa_tuto
    conda install pytorch torchvision torchaudio -c pytorch
    conda install matplotlib pandas scikit-learn

## Usage

-  Train the TinyVGG model on Cifar-10 with standard training and using protections 

Run:

    python3 train_tinyvgg.py


-  Train the CNN model on Cifar-10 with weight clipping and randBET. Change the line #6 and #7 of the *train_tinyvgg.py* script. 

    >clipping_value = 0.1
    >randbet = 1

Then, again: 

    python3 train_tinyvgg.py

-  The models are saved on: ./save/tinyvgg_quan/ (*model_best.pth.tar*) 
-  Process the Bit-Flip Attack on the standard trained model and the *protected* one.

Run:

    python3 bfa_tinyvgg.py


and changing the same parameters (*clipping_value* and *randbet*) as for the training step. 

-  Logs of the 5 attacks are stored in ./save/tinyvgg_quan/

-  Print the evolution of the accuracing vs. the bit-flips. The output image is stored as *tinyvgg_accuracy_vs_bfa.png* (root directory)

Run: 

    python3 printing_tools_tinyvgg.py

NB: Check the printing tool script to change the directories you want to handle in your figure. 

An example for ResNet20 on CIFAR10: 

![](accuracy_vs_bfa.png)


## Tips if you're working on Windows with Anaconda

-  You need to replace **python3** with **python** in files **train_{your_model}.py** and **bfa_{your_model}.py**.

-  You need to add the following lines of code at the end of the files **train_{your_model}.py** and **bfa_{your_model}.py** to redirect the outputs to the console.


```python
with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True) as process:
    for line in process.stdout:
        print(line, end='')
    for line in process.stderr:
        print(line, end='')
```
