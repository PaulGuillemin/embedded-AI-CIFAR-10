import torch
import torch.nn.functional as F

import numpy as np
import random

from models.quantization import quan_Conv2d, quan_Linear


def divide_gradients_by_two(model):
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data /= 2
    return model
    


def transfer_weights(model_source, model_target):
    assert model_source.__class__ == model_target.__class__,"Models are different"
    
    for source_param, target_param in zip(model_source.parameters(), model_target.parameters()):
    
        target_param.data.copy_(source_param.data)
        
    
def apply_global_bit_flips(model, flip_percentage, layer_limit=None):

    
    if layer_limit == -1: 
      layer_limit = None
    
    concatenated_weights_matrix = torch.cat(concatenate_weights(model, layer_limit=layer_limit))
    
    total_elements = concatenated_weights_matrix.numel()
    num_flips = int(total_elements * (flip_percentage / 100))
    
    #print(f"Nombre de bit-flips: {num_flips}/{total_elements}")

    indices_to_flip = random.sample(range(total_elements), num_flips)
    
    
    
    for idx in indices_to_flip:
      value = concatenated_weights_matrix[idx]
      
      flipped_value = apply_bit_flip(int(value))
      #print(f"Initial value: {value} ---- flipped value: {flipped_value}")

      concatenated_weights_matrix[idx] = float(flipped_value)

    start_idx = 0
    flipped_matrices = []
    idx_layer = 0

    for name, m in model.named_modules():
      if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
      
        if layer_limit is not None and idx_layer >= layer_limit:
          break
    
        num_elements = m.weight.data.numel()

        # Extraire la partie de la matrice combin�e correspondant � la matrice actuelle
        end_idx = start_idx + num_elements
        m.weight.data = concatenated_weights_matrix[start_idx:end_idx].reshape(m.weight.data.shape).clone()*m.step_size.item()
        start_idx = end_idx
        
        idx_layer = idx_layer + 1

    return model





def apply_bit_flip(signed_integer_8bit):
        
    bit_to_flip = 7
    
    flipped_integer = signed_integer_8bit ^ (1 << bit_to_flip)
    if bit_to_flip == 7:
        flipped_integer = -(flipped_integer ^ 0xFF) - 1
    
    return flipped_integer
    

def concatenate_weights(model, layer_limit=None):
    concatenated_weights = []
    idx_layer = 0
    for name, m in model.named_modules():
      if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
      
        if layer_limit is not None and idx_layer >= layer_limit:
          break
          
        #print(f"Layer name: {name}")
      
      
        weights_matrix = m.weight.data.reshape(-1).cpu()
        weight_quan = quantif_randbet(weights_matrix.cpu(), m.step_size, m.half_lvls) #* m.step_size
        concatenated_weights.append(weight_quan) 
        idx_layer = idx_layer + 1           
            
    return concatenated_weights
    
    
def quantif_randbet(input, step_size, half_lvls):
    # ctx is a context object that can be used to stash information
    # for backward computation
    
    output = F.hardtanh(input,
                        min_val=-half_lvls * step_size.item(),
                        max_val=half_lvls * step_size.item())

    output = torch.round(output / step_size.item())
    return output


def count_differences(model_original, model_modified):
    # Initialiser le compteur de diff�rences
    different_indices = set()

    # It�rer sur les param�tres des deux mod�les
    for param_original, param_modified in zip(model_original.parameters(), model_modified.parameters()):
        # Trouver les indices o� les poids diff�rent
        
        print(f"{param_original.data.reshape(-1)[:20]}")
        print(f"{param_modified.data.reshape(-1)[:20]}")
        
        indices_diff = torch.nonzero(param_original.data.reshape(-1) != param_modified.data.reshape(-1))
        #print(indices_diff[:10])
        # Ajouter les indices � l'ensemble
        different_indices.update(set(indices_diff))
        print(f"difference: {len(indices_diff)}")
        #print(f"diff�rence:{len(different_indices)}")
        print("#####")
    


