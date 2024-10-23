import json
import shutil
from pathlib import Path

import numpy as np
import torch

from training.tensegrity_gnn_training_engine import TensegrityGNNTrainingEngine


def train():
    # torch.autograd.set_detect_anomaly(True)
    np.set_printoptions(precision=64)
    config_file_path = "training/configs/3_bar_train_config.json"
    with open(config_file_path, 'r') as j:
        config_file = json.load(j)

    num_steps = [1, 2, 4, 8]
    epochs = [200, 100, 50, 25]
    learning_rates = [1e-5, 1e-6, 1e-7, 1e-8]
    batch_sizes = [128, 128, 128, 128]
    load_sim = [False, True, True, True]
    eval_steps = [20, 10, 5, 5]

    params = list(zip(num_steps, epochs, learning_rates, load_sim, batch_sizes, eval_steps))
    for n, e, lr, load, batch_size, eval_step in params[:]:
        config_file['num_steps_fwd'] = n
        config_file['optimizer_params']['lr'] = lr
        config_file['load_sim'] = load
        config_file['batch_size'] = batch_size

        trainer = TensegrityGNNTrainingEngine(config_file, torch.nn.MSELoss(), 0.01)
        trainer.EVAL_STEPSIZE = eval_step

        trainer.to('cuda')
        trainer.run(e)

        output_dir = Path(config_file['output_path'])
        try:
            shutil.copy(output_dir / "best_loss_model.pt", output_dir / f"{n}_steps_best_loss_model.pt")
            shutil.copy(output_dir / "best_rollout_model.pt", output_dir / f"{n}_steps_best_rollout_model.pt")
        except:
            print("No best_rollout_model")


if __name__ == '__main__':
    train()
