import matplotlib
matplotlib.use('Agg',force=True)
import matplotlib.pyplot as plt
import argparse
import os
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras as k
from PendulumData import Pendulum
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

def generate_pendulum_filter_dataset(pendulum, num_seqs, seq_length, seed):
    obs, targets, _, _ = pendulum.sample_data_set(num_seqs, seq_length, full_targets=False, seed=seed)
    obs, _ = pendulum.add_observation_noise(obs, first_n_clean=5, r=0.2, t_ll=0.0, t_lu=0.25, t_ul=0.75, t_uu=1.0)
    obs = np.expand_dims(obs, -1)
    return obs.astype(np.float32), targets.astype(np.float32)


# Implement Encoder and Decoder hidden layers
class PendulumStateEstemGIN(GIN):

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
        return [k.layers.Dense(units=10, activation=k.activations.relu)]

    def build_var_decoder_hidden(self):
        return [k.layers.Dense(units=10, activation=k.activations.relu)]
     
    


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
        # Generate Data
        pend_params = Pendulum.pendulum_default_params()
        pend_params[Pendulum.FRICTION_KEY] = 0.1
        data = Pendulum(24, observation_mode=Pendulum.OBSERVATION_MODE_LINE,
                        transition_noise_std=0.1,
                        observation_noise_std=1e-5,
                        seed=0,
                        pendulum_params=pend_params)

        train_obs, train_targets = generate_pendulum_filter_dataset(data, 1000, 75, np.random.randint(100000000))
        valid_obs, valid_targets = generate_pendulum_filter_dataset(data, 100, 75, np.random.randint(100000000))
        test_obs, test_targets = generate_pendulum_filter_dataset(data, 100, 75, np.random.randint(10000000))

        #Build Model
        gin = PendulumStateEstemGIN(observation_shape=train_obs.shape[-3:], 
                                latent_observation_dim = configs[key]["lod"], 
                                latent_state_dim = configs[key]["lsd"], 
                                output_dim=train_targets.shape[-1], num_basis=configs[key]["Num_Bases"],
                                never_invalid=True, cell_type = configs[key]["cell"], Qnetwork= configs[key]["QNetwork"], 
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

        Training_Loss = gin.training( gin, train_obs, train_targets, valid_obs, valid_targets,
                                    test_obs, test_targets, epochs, batch_size,
                                    x_epoch, record, fig, ax0, draw_fig= bool(configs[key]["draw_fig"]))
        Test_Loss = gin.testing( gin, test_obs, test_targets, batch_size)

if __name__ == '__main__':
	main()
