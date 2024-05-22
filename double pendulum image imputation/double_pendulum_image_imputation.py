import matplotlib
matplotlib.use('Agg',force=True)
import matplotlib.pyplot as plt
import argparse
import os
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras as k
from DoublePendulum import DoublePendulum
from GIN import GIN
from LayerNormalizer import LayerNormalizer

def read_json(path):
    with open(path) as f:
        d = json.load(f)
    f.close()
    return d

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs.json")
    return parser.parse_args()

def generate_imputation_data_set( num_seqs, seq_length, seed):
    pendulum = DoublePendulum(num_seqs, seq_length, 24, seed)
    obs, _ = pendulum.datagen()
    obs = np.expand_dims(obs, -1)
    targets = obs.copy()

    rs = np.random.RandomState(seed=seed)
    obs_valid = rs.rand(num_seqs, seq_length, 1) < 0.9
    obs_valid[:, :20] = True
    print("Fraction of Valid Images:", np.count_nonzero(obs_valid) / np.prod(obs_valid.shape))
    obs[np.logical_not(np.squeeze(obs_valid))] = 0

    return obs.astype(np.float32), obs_valid, targets.astype(np.float32)


# Implement Encoder and Decoder hidden layers
class DoublePendulumImageImputationGIN(GIN):

    def build_encoder_hidden(self):
        return [
            # 1: Conv Layer
            k.layers.Conv2D(12, kernel_size=5, padding="same"),
            LayerNormalizer(),
            k.layers.Activation(k.activations.relu),
            k.layers.MaxPool2D(2, strides=2),
            # 2: Conv Layer
            k.layers.Conv2D(12, kernel_size=3, padding="same", strides=2),
            LayerNormalizer(),
            k.layers.Activation(k.activations.relu),
            k.layers.MaxPool2D(2, strides=2),
            k.layers.Flatten(),
            # 3: Dense Layer
            k.layers.Dense(30, activation=k.activations.relu)]

    def build_decoder_hidden(self):
        return [
            k.layers.Dense(144, activation=k.activations.relu),
            k.layers.Lambda(lambda x: tf.reshape(x, [-1, 3, 3, 16])),

            k.layers.Conv2DTranspose(16, kernel_size=5, strides=4, padding="same"),
            LayerNormalizer(),
            k.layers.Activation(k.activations.relu),

            k.layers.Conv2DTranspose(12, kernel_size=3, strides=2, padding="same"),
            LayerNormalizer(),
            k.layers.Activation(k.activations.relu)
        ]
    


def main():
    args = parse_args()

    # config_path = "./config.json"
    config_path = args.config
    configs = read_json(config_path)
    

    for key in configs.keys():
        result_path = "./results/" + key
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        
        config_name = ""
        for elems in configs[key].keys():
            config_name += elems + "_" + str(configs[key][elems]) + "\n"
        with open(result_path + '/config.txt', 'w') as config_wr:
            config_wr.write(config_name)
        config_wr.close()

        print("running config: {}".format(config_name))


        train_obs, train_obs_valid, train_targets = generate_imputation_data_set( 1000, 75, seed=42)
        valid_obs, valid_obs_valid, valid_targets = generate_imputation_data_set( 100, 75, seed=42)
        test_obs, test_obs_valid, test_targets = generate_imputation_data_set( 100, 75, seed=23541)
        #Build Model
        gin = DoublePendulumImageImputationGIN(observation_shape=train_obs.shape[-3:],
                                    latent_observation_dim=configs[key]["lod"],
                                    latent_state_dim = configs[key]["lsd"],
                                    output_dim=train_targets.shape[-3:], num_basis=configs[key]["Num_Bases"],
                                    never_invalid=False,
                                    cell_type = configs[key]["cell"], 
                                    Qnetwork= configs[key]["QNetwork"], 
                                    Smoothing=configs[key]["Smoothing"], 
                                    USE_CONV= bool(configs[key]["Use_Conv_Covar"]), 
                                    USE_MLP_AFTER_KGGRU = bool(configs[key]["USE_MLP_AFTER_KGGRU"]),
                                    KG_Units = configs[key]["KG_Units"],
                                    Xgru_Units =configs[key]["Xgru_Units"],
                                    Fgru_Units =configs[key]["Fgru_Units"],
                                    KG_InputSize = configs[key]["KG_InputSize"],
                                    Xgru_InputSize = configs[key]["Xgru_InputSize"],
                                    Fgru_InputSize = configs[key]["Fgru_InputSize"],
                                    lr = configs[key]["lr"],
                                    lr_decay = configs[key]["lr_decay"],
                                    lr_decay_it = configs[key]["lr_decay_iteration"],
                                    result_path = result_path)



        # Plot Loss
        if bool(configs[key]["draw_fig"]) == True:
            x_epoch = []
            record = {'train_loss':[], 'train_err':[], 'test_loss':[], 'test_err':[]}
            fig = plt.figure()
            ax0 = fig.add_subplot(121, title="loss")
        # Train Model
        epochs, batch_size = configs[key]["epochs"], configs[key]["batch_size"]

        Training_Loss = gin.training( gin, train_obs, train_obs_valid, train_targets, valid_obs, valid_obs_valid, valid_targets,
                                    test_obs, test_obs_valid, test_targets, epochs, batch_size,
                                    x_epoch, record, fig, ax0, draw_fig= bool(configs[key]["draw_fig"]))
        Test_Loss = gin.testing( gin, test_obs, test_obs_valid, test_targets, batch_size)

if __name__ == '__main__':
	main()


