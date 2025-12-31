from time import time
from helpers.BaseRunner import BaseRunner

class GTERunner(BaseRunner):
    @staticmethod
    def parse_runner_args(parser):
        return BaseRunner.parse_runner_args(parser)
    
    def train(self, data_dict):
        model = data_dict['train'].model
        
        print("\n" + "="*50)
        print("GTE Model (No Training Required)")
        print("="*50)
        
        start_time = time()
        model._ensure_propagation()
        print(f"Propagation completed in {time()-start_time:.2f}s")
        
        self.epoch = 1
        self.early_stop = 1
        super().train(data_dict)
    
    def fit(self, dataset, epoch=-1):
        return 0.0