




from src.traininig.train_weak import train_model
from src.traininig.fine_tune import train_model as finetune_model

from src.postprocessing.postprocess import postprocess_model

 # train source model
config_source = {"dataset": "", # name of the source dataset
                 "latent_dim": 10, # number of latent dimensions
                 "random_seed": 42,
                 "method" : "EFFICIENTVAEWEAK",
                 "n_filters": 128,
                 "beta": 1., # VAE regularize
                 "warm_up_iterations": 50000,
                 "batch_size": 64,
                 "lr": 0.001,
                 "wd": 0.0, # weight decay
                 "iterations": 300000, # iterations to train the model
                 "factor_idx": list(range(0, 4)), # list of the FoVs to train the model on
}

postprocessing_config = {}

output_directory = "path where to save source model"
train_model(output_directory, config_source)
postprocess_model(output_directory, postprocessing_config)

# transfer to target model
config_transfer = {"dataset": "", # name of the target dataset
                 "latent_dim": 10, # number of latent dimensions
                 "random_seed": 42,
                 "method" : "EFFICIENTVAE",
                 "n_filters": 128,
                 "beta": 1., # VAE regularize
                 "warm_up_iterations": 50000,
                 "batch_size": 64,
                 "lr": 0.0001,
                 "wd": 0.0, # weight decay
                 "iterations": 10000, # iterations to finetune the model
                 "factor_idx": list(range(0, 4)), # list of the FoVs to train the model on
}
output_target_directory = "path where to save transferred model"
finetune_model(output_target_directory, config_transfer)  # finetune target model
postprocess_model(output_target_directory, config_transfer) # extract representation to evaluate