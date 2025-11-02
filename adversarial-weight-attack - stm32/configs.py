class args():
    

    def __init__(self, opt) -> None:
    
      self.data_path = "./dataset"
      self.arch = opt.arch
      
      config = {
          'tinyvgg_quan': {
              'dataset': 'cifar10',
              'epochs': 40,
              'start_epoch': 0,
              'attack_sample_size': 100,
              'test_batch_size': 100,
              'optimizer': "SGD",
              'schedule': [25, 40],
              'gammas': [0.1, 0.1],
              'workers': 4,
              'ngpu': 0,
              'gpu_id': 0,
              'print_freq': 100,
              'decay': 0.0003,
              'momentum': 0.9,
              'limit_layer': 7, #Number of layer implicated in randbet
              'randbet_coeff': 10,
              'k_top': 100
          },    

          'resnet20_quan': {
              'dataset': 'cifar10',
              'epochs': 40,
              'start_epoch': 0,
              'attack_sample_size': 100,
              'test_batch_size': 100,
              'optimizer': "SGD",
              'schedule': [25, 40],
              'gammas': [0.1, 0.1],
              'workers': 4,
              'ngpu': 0,
              'gpu_id': 0,
              'print_freq': 100,
              'decay': 0.0003,
              'momentum': 0.9,
              'limit_layer': 7,
              'randbet_coeff': 10,
              'k_top': 100
          },
          
          'cnn_quan': {
              'dataset': 'mit-bih',
              'epochs': 20,
              'start_epoch': 0,
              'attack_sample_size': 128,
              'test_batch_size': 128,
              'optimizer': "Adam",
              'schedule': [25, 40],
              'gammas': [0.1, 0.1],
              'workers': 4,
              'ngpu': 0,
              'gpu_id': 0,
              'print_freq': 100,
              'decay': 0.0003,
              'momentum': 0.9,
              'limit_layer': -1,
              'randbet_coeff': 5,
              'k_top': 20  # Depends on the minimal number of parameters in a layer
          }
      }

      
      
      model_config = config[self.arch]
      
      for key, value in model_config.items():
        setattr(self, key, value)
        
        
      self.randbet = opt.randbet
      self.limit_layer = opt.limit_layer
      self.randbet_coeff = opt.randbet_coeff
      self.clipping_coeff = opt.clipping_coeff
      self.learning_rate = opt.learning_rate
      
      self.manualSeed = opt.manualSeed
      
      
      if self.clipping_coeff != 0.0 and self.randbet == 1:
          label_info = f"randbet_{self.clipping_coeff}_{self.learning_rate}_{self.randbet_coeff}_{self.limit_layer}"
      elif self.clipping_coeff != 0.0 and self.randbet == 0:
          label_info = f"clipping_{self.clipping_coeff}_{self.learning_rate}"
      else:
          label_info = f"nominal_{self.learning_rate}"
    
      print(label_info)
    
      self.save_path = f"./save/{self.arch}/{label_info}"
      
      
      self.enable_bfa = opt.enable_bfa if hasattr(opt, 'enable_bfa') else False
      
      self.resume = ""
      self.quan_bitwidth = None
      self.reset_weight = False
      self.evaluate = False
      
      self.n_iter = 30
      
      
      
      
      if self.enable_bfa:

          self.reset_weight = True 
          self.save_path = f"./save/{self.arch}/{label_info}/results/{self.manualSeed}"
          self.resume = f"./save/{self.arch}/{label_info}/model_best.pth.tar" 
          
          self.fine_tune =True
          self.evaluate = True
          
          
      self.model_only = opt.model_only if hasattr(opt, 'model_only') else False
      self.random_bfa = opt.random_bfa if hasattr(opt, 'random_bfa') else False
      

      
      
      
    def _get_kwargs(self):
    
      return [(k, v) for k, v in self.__dict__.items()]
       
    
    
