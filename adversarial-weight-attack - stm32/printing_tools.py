from __future__ import division
from __future__ import absolute_import

import os, sys, shutil, time, random
import argparse

import numpy as np
import csv
import matplotlib.pyplot as plt

bestlength = 30
max_index = 30
seeds = [6213, 4258, 3666, 758, 5555]


def generate_file_paths(folder, seeds):
    file_paths = [f"./save/resnet20_quan/{folder}/results/{seed}/attack_profile_{seed}.csv"
                  for seed in seeds]
    return file_paths

def read_csv_data(files):
    list1_r, list2_r = [], []
    for fil in files:
        min_list_1, min_list_2 = [], []
        with open(fil, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                min_list_1.append(int(row[1]))
                min_list_2.append(float(row[6]))
        list1_r.append(min_list_1)
        list2_r.append(min_list_2)
    return list1_r, list2_r

def resize_list(liste, bestlength):
    lastvalue = liste[-1]
    extend_v = [lastvalue] * (bestlength - len(liste))
    return liste + extend_v
    
    
def main():

  folders = ['clipping_0.1_0.1', 'nominal_0.1', 'nominal_0.01', 'randbet_0.1_0.1_10_-1']
  colors = ['green', 'red', 'blue', 'yellow']
  labels = ['Clipping - 0.1', 'Nominal 0.1', 'Nominal 0.01', 'Randbet w clipping 0.1']
  
  mean_values = []
  min_values = []
  max_values = []
  
  for folder in folders:
      print(folder)
      liste_file = generate_file_paths(folder, seeds)
      
      list1_r, list2_r = read_csv_data(liste_file)
      list2_r = [resize_list(x, bestlength) for x in list2_r]
      list2_r = np.array(list2_r)
      
      #print(list2_r)
      
      liste_var = np.std(list2_r, axis=0)
      mean_list = np.mean(list2_r, axis=0)
      
      min_meanstd = mean_list - np.abs(liste_var)
      max_meanstd = mean_list + np.abs(liste_var)
      
      mean_values.append(mean_list)
      min_values.append(min_meanstd)
      max_values.append(max_meanstd)
  
  array_x = np.arange(1, bestlength + 1)
  
  plt.grid(alpha=0.2, linestyle='--')
  plt.yticks(np.arange(0, 101, 10))
  plt.xticks(np.arange(0, max_index + 1, max_index / 10))
  
  for i in range(len(folders)):
      plt.plot(array_x[:max_index], mean_values[i].T[:max_index], color=colors[i], label=labels[i])
      plt.fill_between(array_x[:max_index], min_values[i][:max_index], max_values[i][:max_index], color=colors[i], alpha=0.1)
  
  plt.legend()
  plt.xlabel('# bit-flips')
  plt.ylabel('Accuracy (%)')
  plt.savefig('./accuracy_vs_bfa.png')
  
  plt.clf()

if __name__ == '__main__':
    main()
