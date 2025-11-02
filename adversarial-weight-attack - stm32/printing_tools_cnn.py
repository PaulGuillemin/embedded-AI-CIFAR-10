from __future__ import division
from __future__ import absolute_import

import os, sys, shutil, time, random
import argparse

import numpy as np
import csv
import matplotlib.pyplot as plt

bestlength = 30
max_index = 30
seeds = [5555,758,3666,4258,6213]


def generate_file_paths(folder, seeds):
    file_paths = [f"./save/cnn_quan/{folder}/results/{seed}/attack_profile_{seed}.csv" for seed in seeds]
    return file_paths

def read_csv_data(files):
    list1_r, list2_r = [], []
    for fil in files:
        min_list_1 = []
        min_list_2 = []
        try:
            with open(fil, 'r') as file:
                reader = csv.reader(file)
                next(reader)
                rows = list(reader)
        except Exception as e:
            print(f"Erreur en ouvrant {fil} : {e}")
            list1_r.append(min_list_1)
            list2_r.append(min_list_2)
            continue

        if len(rows) == 0:
            list1_r.append(min_list_1)
            list2_r.append(min_list_2)
            continue

        try:
            nominal = float(rows[0][6]) + float(rows[0][7])
        except Exception as e:
            try:
                nominal = float(rows[0][6])
            except Exception as e2:
                nominal = 0.0
                print(f"Impossible de lire nominal dans {fil} : {e}; fallback à 0.0")

        try:
            min_list_1 = [0] + [int(r[1]) for r in rows]
        except Exception:
            min_list_1 = [0] + [r[1] for r in rows]

        accs = []
        for r in rows:
            try:
                accs.append(float(r[6]))
            except Exception:
                accs.append(np.nan)
        min_list_2 = [nominal] + accs

        list1_r.append(min_list_1)
        list2_r.append(min_list_2)
    return list1_r, list2_r

def resize_list(liste, bestlength):
    if len(liste) == 0:
        return [0.0] * bestlength
    lastvalue = liste[-1]
    if len(liste) >= bestlength:
        return liste[:bestlength]
    extend_v = [lastvalue] * (bestlength - len(liste))
    return liste + extend_v
    
    
def main():

  folders = ['nominal_0.01','nominal_0.001']
  colors = ['green','red']
  labels = ['Nominal 0.01','Nominal 0.001']
  
  mean_values = []
  min_values = []
  max_values = []
  
  for folder in folders:
      print(folder)
      liste_file = generate_file_paths(folder, seeds)
      
      list1_r, list2_r = read_csv_data(liste_file)
      list2_r = [resize_list(x, bestlength) for x in list2_r]
      list2_r = np.array(list2_r)
      
      # calculs
      liste_var = np.std(list2_r, axis=0)
      mean_list = np.nanmean(list2_r, axis=0)
      # intervalle mean +/- std
      min_meanstd = mean_list - liste_var
      max_meanstd = mean_list + liste_var
      
      mean_values.append(mean_list)
      min_values.append(min_meanstd)
      max_values.append(max_meanstd)
  
  array_x = np.arange(0, bestlength)
  
  plt.grid(alpha=0.2, linestyle='--')
  plt.yticks(np.arange(0, 101, 10))
  step = max(1, int(max_index // 10))
  plt.xticks(np.arange(0, max_index + 1, step))
  
  for i in range(len(folders)):
      plt.plot(array_x[:max_index], mean_values[i].T[:max_index], color=colors[i], label=labels[i])
      plt.fill_between(array_x[:max_index], min_values[i][:max_index], max_values[i][:max_index], color=colors[i], alpha=0.1)
  
  plt.legend()
  plt.xlabel('# bit-flips (0 = nominal)')
  plt.ylabel('Accuracy (%)')
  plt.savefig('./cnn_accuracy_vs_bfa.png')
  
  plt.clf()

if __name__ == '__main__':
    main()

