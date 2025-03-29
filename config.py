__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '3.6'
__status__ = "Research"
__date__ = "1/12/2018"
__license__= "MIT License"

from torch.autograd import Variable


class HParameters:

    def __init__(self):
        # Original parameters
        self.verbose = False
        self.use_cuda = False
        self.cuda_device = 0
        self.max_summary_length = 0.15
        self.l2_req = 0.00001
        self.lr_epochs = [0]
        self.lr = [0.00005]
        self.epochs_max = 50
        self.train_batch_size = 1

        # New hyperparameters for improved training
        self.num_heads = 8  # Number of attention heads
        self.dropout = 0.5  # Dropout rate
        self.temporal_weight = 0.1  # Weight for temporal consistency loss
        self.diversity_weight = 0.15  # Weight for diversity loss
        self.attention_hidden_size = 1024  # Hidden size for attention
        self.early_stopping_patience = 5  # Epochs to wait before early stopping
        self.warmup_epochs = 5  # Number of epochs for learning rate warmup
        self.min_lr = 1e-4  # Minimum learning rate
        # self.min_lr = 1e-4  # Minimum learning rate
        self.clip_grad_norm = 1.0  # Gradient clipping threshold
        
        # Focus only on SumMe dataset
        self.datasets = ['datasets/eccv16_dataset_summe_google_pool5.h5']
        self.splits = ['splits/summe_splits.json']
        
        self.output_dir = 'experiments/summe_enhanced'
        self.root = ''

    def get_dataset_by_name(self, dataset_name):
        for d in self.datasets:
            if dataset_name in d:
                return [d]
        return None

    def load_from_args(self, args):
        for key in args:
            val = args[key]
            if val is not None:
                if hasattr(self, key) and isinstance(getattr(self, key), list):
                    val = val.split()
                setattr(self, key, val)

    def __str__(self):
        vars = [attr for attr in dir(self) if not callable(getattr(self,attr)) and not (attr.startswith("__") or attr.startswith("_"))]
        info_str = ''
        for i, var in enumerate(vars):
            val = getattr(self, var)
            info_str += '['+str(i)+'] '+var+': '+str(val)+'\n'
        return info_str


if __name__ == "__main__":

    # Tests
    hps = HParameters()
    print(hps)

    args = {'root': 'root_dir',
            'datasets': 'set1,set2,set3',
            'splits': 'split1, split2',
            'new_param_float': 1.23456
            }

    hps.load_from_args(args)
    print(hps)
