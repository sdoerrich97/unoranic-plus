"""
xAILAB Bamberg
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Training.

@author: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
"""

# Import packages
import argparse
from pathlib import Path
import time
import wandb
import torch
from copy import deepcopy
from timm.optim import optim_factory
import torch.optim as optim
import yaml
import numpy as np
from tqdm import tqdm

# Import own scripts
from data_loader import DataLoaderCustom
from ..models.unoranic import Unoranic
from ..models.unoranic_plus import UnoranicPlus

from ..utils import LearningRate


class Trainer:
    def __init__(self, configuration):
        """
        Initialize the trainer.
        """

        # Read out the parameters of the training run
        self.dataset = configuration['dataset']
        self.data_path = Path(configuration['data_path'])
        self.output_path = Path(configuration['output_path'])
        self.architecture = configuration['architecture']
        self.resume_training = configuration['resume_training']['resume']
        self.seed = configuration['seed']
        self.device = torch.device(configuration['device'])
        self.hp_optimization = configuration['hp_optimization']

        # Read out the hyperparameters of the training run
        self.starting_epoch = configuration['starting_epoch']
        self.epochs = configuration['epochs']
        self.img_size = configuration['img_size']
        self.in_channel = configuration['in_channel']
        self.patch_size = configuration['patch_size']
        self.latent_dim = configuration['latent_dim']
        self.depth = configuration['depth']
        self.num_heads = configuration['num_heads']
        self.mlp_ratio = configuration['mlp_ratio']
        self.norm_layer = configuration['norm_layer']
        self.batch_size = configuration['batch_size']
        self.tuple_size = configuration['tuple_size']
        self.dropout = configuration['dropout']

        self.learning_rate = configuration['optimizer']['learning_rate'] * self.batch_size / 256
        self.learning_rate_min = configuration['optimizer']['learning_rate_min']
        self.first_momentum = configuration['optimizer']['first_momentum']
        self.second_momentum = configuration['optimizer']['second_momentum']
        self.weight_decay = configuration['optimizer']['weight_decay']
        self.warmup_epochs = configuration['optimizer']['warmup_epochs']

        # Create the path to where the output shall be stored and initialize the logger
        self.output_path = self.output_path / f"{self.latent_dim}" / self.dataset / self.architecture
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Enable the use of cudnn
        torch.backends.cudnn.benchmark = True

        # Fix the hyperparameters for the unoranic model for comparability with the original publication
        if self.architecture == 'unoranic':
            self.latent_dim = 256
            self.epochs = 1000
            self.optimizer = 'adam'
            self.learning_rate = 0.0001
            self.first_momentum = 0.95
            self.second_momentum = 0.999
            self.scheduler_lr_base = 0.0001
            self.scheduler_lr_max = 0.01
            self.scheduler_step_up = 2000

        # Initialize the dataloader
        print("\tInitializing the dataloader...")
        self.data_loader = DataLoaderCustom(self.dataset, self.data_path, self.img_size, self.batch_size,
                                            self.tuple_size, train=True)
        self.train_loader = self.data_loader.get_train_loader()
        self.val_loader = self.data_loader.get_val_loader()

        # Initialize the model
        print("\tInitializing the model ...")
        if self.architecture == 'unoranic':
            self.model = Unoranic(self.img_size, self.in_channel, self.latent_dim, self.dropout)

        elif self.architecture == 'unoranic+':
            self.model = UnoranicPlus(self.img_size, self.in_channel, self.patch_size, self.latent_dim, self.depth,
                                      self.num_heads, self.mlp_ratio, self.norm_layer)

        else:
            raise SystemExit

        self.model.to(self.device)
        self.model.requires_grad_(True)

        # Initialize the optimizer and learning rate scheduler
        print("\tInitializing the optimizer and lr scheduler...")
        if self.architecture == 'unoranic':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                              betas=(self.first_momentum, self.second_momentum))
            self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.scheduler_lr_base,
                                                         max_lr=self.scheduler_lr_max,
                                                         step_size_up=len(self.train_loader) * 5,
                                                         mode='triangular2', cycle_momentum=False)

        else:
            param_groups = optim_factory.param_groups_weight_decay(self.model, self.weight_decay)
            self.optimizer = torch.optim.AdamW(param_groups, lr=self.learning_rate, betas=(self.first_momentum,
                                                                                           self.second_momentum))

        # Create variables to store the best performing model
        print("\tInitializing helper variables...")
        self.best_loss = np.inf
        self.best_epoch = self.starting_epoch
        self.best_model = deepcopy(self.model)

        self.lambda_reconstruction = 1
        self.lambda_robustness = 1

    def train(self):
        """
        Run the training.
        """

        # Start code
        start_time = time.time()

        # Empty the unused memory cache
        print("\tEmptying the unused memory cache...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Train the classifier for the specified number of epochs
        print(f"\tRun the training for {self.epochs} epochs...")
        for epoch in range(self.starting_epoch, self.epochs + 1):
            # Stop the time
            start_time_epoch = time.time()
            print(f"\t\tEpoch {epoch} of {self.epochs}:")

            # Run the training and validation for the current epoch
            # Run the training
            self.run_iteration('Train', self.train_loader, epoch)

            # Run the validation
            with torch.no_grad():
                self.run_iteration('Val', self.val_loader, epoch)

            # Save the checkpoint
            print(f"\t\t\tSaving the checkpoint...")
            if not self.hp_optimization:
                # Save the checkpoint
                self.save_model(epoch)

            # Stop the time for the epoch
            end_time_epoch = time.time()
            hours_epoch, minutes_epoch, seconds_epoch = self.calculate_passed_time(start_time_epoch, end_time_epoch)

            print("\t\t\tElapsed time for epoch: {:0>2}:{:0>2}:{:05.2f}".format(hours_epoch, minutes_epoch,
                                                                                seconds_epoch))

        # Stop the time
        end_time = time.time()
        hours, minutes, seconds = self.calculate_passed_time(start_time, end_time)

        print("\t\t\tElapsed time for epoch: {:0>2}:{:0>2}:{:05.2f}".format(hours, minutes, seconds))

    def run_iteration(self, mode, data_loader, epoch):
        """
        Run one epoch for the current model.

        :param mode: Train or validation mode.
        :param data_loader: Data loader for the current mode.
        :param epoch: Current epoch.
        """

        # Initialize a progress bar
        pbar = tqdm(total=len(data_loader), bar_format='{l_bar}{bar}', ncols=80, initial=0, position=0, leave=False)
        pbar.set_description(f'\t\t\t{mode}')

        # Set the model according to the mode
        if mode == 'Train':
            self.model.train()

        else:
            self.model.eval()

        # Initialize the loss values
        total_loss, reconstruction_loss, robustness_loss = 0, 0, 0
        mse_input, mse_anatomy, mse_characteristics = 0, 0, 0
        ssim_input, ssim_anatomy, ssim_characteristics = 0, 0, 0
        psnr_input, psnr_anatomy, psnr_characteristics = 0, 0, 0
        consistency_loss_anatomy, consistency_loss_characteristics = 0, 0
        mean_cos_dist_anatomy, mean_cos_dist_characteristics = 0, 0

        # Iterate through all samples
        for i, (X_orig, _, X, X_anatomy_tuple, X_characteristics_tuple, _) in enumerate(data_loader):
            # Map the input to the respective device
            X_orig = X_orig.to(self.device)
            X = X.to(self.device)

            # Run the input through the model
            if self.architecture == 'unoranic':
                # Create the tuple for the anatomy and characteristic branch
                X_anatomy, X_characteristics = [X.clone().to(self.device)], [X.clone().to(self.device)]

                for X_a, X_c in zip(X_anatomy_tuple, X_characteristics_tuple):
                    X_anatomy.append(X_a.to(self.device))
                    X_characteristics.append(X_c.to(self.device))

                (mse_input_batch, mse_anatomy_batch, mse_characteristics_batch), \
                (ssim_input_batch, ssim_anatomy_batch, ssim_characteristics_batch), \
                (psnr_input_batch, psnr_anatomy_batch, psnr_characteristics_batch), \
                (consistency_loss_anatomy_batch, consistency_loss_characteristics_batch), \
                (mean_cos_dist_anatomy_batch, mean_cos_dist_characteristics_batch), _ = self.model(X_orig, X, X_anatomy, X_characteristics)

            else:
                (mse_input_batch, mse_anatomy_batch, mse_characteristics_batch), \
                (ssim_input_batch, ssim_anatomy_batch, ssim_characteristics_batch), \
                (psnr_input_batch, psnr_anatomy_batch, psnr_characteristics_batch), \
                (consistency_loss_anatomy_batch, consistency_loss_characteristics_batch), \
                (mean_cos_dist_anatomy_batch, mean_cos_dist_characteristics_batch), _ = self.model(X_orig, X)

            # Create the final losses as a combination of the individual losses
            if self.architecture == 'unoranic':
                robustness_loss_batch = consistency_loss_anatomy_batch
                reconstruction_loss_batch = mse_input_batch + mse_anatomy_batch
                total_loss_batch = self.lambda_reconstruction * reconstruction_loss_batch + self.lambda_robustness * robustness_loss_batch

            else:
                robustness_loss_batch = consistency_loss_anatomy_batch
                reconstruction_loss_batch = mse_input_batch + mse_anatomy_batch
                total_loss_batch = self.lambda_reconstruction * reconstruction_loss_batch

            # Run the backward pass for train mode
            if mode == 'Train':
                # Zero out the gradients
                self.optimizer.zero_grad()

                # Run the backward pass for the reconstruction of the corrupted input
                total_loss_batch.backward()

                # Optimize the weights
                self.optimizer.step()

                # Adjust the learning rate
                if self.architecture == 'unoranic+':
                    LearningRate.adjust_learning_rate(self.optimizer, i / len(data_loader) + epoch, self.learning_rate,
                                                      self.learning_rate_min, self.epochs, self.warmup_epochs)

            # Append the current loss values to the overall values for the current epoch
            total_loss += total_loss_batch.detach().cpu()
            reconstruction_loss += reconstruction_loss_batch.detach().cpu()
            robustness_loss += robustness_loss_batch.detach().cpu()

            mse_input += mse_input_batch.detach().cpu()
            mse_anatomy += mse_anatomy_batch.detach().cpu()
            mse_characteristics += mse_characteristics_batch.detach().cpu()
            ssim_input += ssim_input_batch.detach().cpu()
            ssim_anatomy += ssim_anatomy_batch.detach().cpu()
            ssim_characteristics += ssim_characteristics_batch.detach().cpu()
            psnr_input += psnr_input_batch.detach().cpu()
            psnr_anatomy += psnr_anatomy_batch.detach().cpu()
            psnr_characteristics += psnr_characteristics_batch.detach().cpu()
            consistency_loss_anatomy += consistency_loss_anatomy_batch.detach().cpu()
            consistency_loss_characteristics += consistency_loss_characteristics_batch.detach().cpu()
            mean_cos_dist_anatomy += mean_cos_dist_anatomy_batch
            mean_cos_dist_characteristics += mean_cos_dist_characteristics_batch

            # Update the progress bar
            pbar.update(1)

        # Update the lr scheduler for unoranic and unoranic+
        if mode == 'Train' and self.architecture == 'unoranic':
            self.scheduler.step()

        # Average the loss values across all batches
        total_loss /= len(data_loader)
        reconstruction_loss /= len(data_loader)
        robustness_loss /= len(data_loader)

        mse_input /= len(data_loader)
        mse_anatomy /= len(data_loader)
        mse_characteristics /= len(data_loader)
        ssim_input /= len(data_loader)
        ssim_anatomy /= len(data_loader)
        ssim_characteristics /= len(data_loader)
        psnr_input /= len(data_loader)
        psnr_anatomy /= len(data_loader)
        psnr_characteristics /= len(data_loader)
        consistency_loss_anatomy /= len(data_loader)
        consistency_loss_characteristics /= len(data_loader)
        mean_cos_dist_anatomy /= len(data_loader)
        mean_cos_dist_characteristics /= len(data_loader)

        # Print the loss values and send them to wandb
        print(f"\t\t\t{mode} Total Loss: {total_loss}")
        print(f"\t\t\t{mode} Reconstruction Loss: {reconstruction_loss}")
        print(f"\t\t\t{mode} Robustness Loss: {robustness_loss}")
        print(f"\t\t\t{mode} MSE Input: {mse_input}")
        print(f"\t\t\t{mode} MSE Anatomy: {mse_anatomy}")
        print(f"\t\t\t{mode} MSE Characteristics: {mse_characteristics}")
        print(f"\t\t\t{mode} SSIM Input: {ssim_input}")
        print(f"\t\t\t{mode} SSIM Anatomy: {ssim_anatomy}")
        print(f"\t\t\t{mode} SSIM Characteristics: {ssim_characteristics}")
        print(f"\t\t\t{mode} PSNR Input: {psnr_input}")
        print(f"\t\t\t{mode} PSNR Anatomy: {psnr_anatomy}")
        print(f"\t\t\t{mode} PSNR Characteristics: {psnr_characteristics}")
        print(f"\t\t\t{mode} Consistency Loss Anatomy: {consistency_loss_anatomy}")
        print(f"\t\t\t{mode} Consistency Loss Characteristics: {consistency_loss_characteristics}")
        print(f"\t\t\t{mode} Mean Cosine Distance Anatomy: {mean_cos_dist_anatomy}")
        print(f"\t\t\t{mode} Mean Cosine Distance Characteristics: {mean_cos_dist_characteristics}")

        wandb.log(
            {
                f"{mode} Total Loss": total_loss,
                f"{mode} Reconstruction Loss": reconstruction_loss,
                f"{mode} Robustness Loss": robustness_loss,
                f"{mode} MSE Input": mse_input,
                f"{mode} MSE Anatomy": mse_anatomy,
                f"{mode} MSE Characteristics": mse_characteristics,
                f"{mode} SSIM Input": ssim_input,
                f"{mode} SSIM Anatomy": ssim_anatomy,
                f"{mode} SSIM Characteristics": ssim_characteristics,
                f"{mode} PSNR Input": psnr_input,
                f"{mode} PSNR Anatomy": psnr_anatomy,
                f"{mode} PSNR Characteristics": psnr_characteristics,
                f"{mode} Consistency Loss Anatomy": consistency_loss_anatomy,
                f"{mode} Consistency Loss Characteristics": consistency_loss_characteristics,
                f"{mode} Mean Cosine Distance Anatomy": mean_cos_dist_anatomy,
                f"{mode} Mean Cosine Distance Characteristics": mean_cos_dist_characteristics,
            }, step=epoch)

        # Store the current best model
        if mode == 'Val':
            if total_loss < self.best_loss:
                self.best_epoch = epoch
                self.best_loss = total_loss
                self.best_model = deepcopy(self.model)

            print(f"\t\t\tCurrent best Total Loss: {self.best_loss}")
            print(f"\t\t\tCurrent best Epoch: {self.best_epoch}")

    def save_model(self, epoch: int, save_idx=50):
        """
        Save the model every save_idx epochs.

        :param epoch: Current epoch.
        :param save_idx: Which epochs to save.
        """

        # Create the checkpoint
        if self.architecture == 'unoranic':
            checkpoint = {
                'model': self.model.state_dict(),
                'encoder_anatomy': self.model.encoder_anatomy.state_dict(),
                'encoder_characteristics': self.model.encoder_characteristics.state_dict(),
                'decoder_anatomy': self.model.decoder_anatomy.state_dict(),
                'decoder': self.model.decoder.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            }

            checkpoint_final = {
                'model': self.model.state_dict(),
                'encoder_anatomy': self.model.encoder_anatomy.state_dict(),
                'encoder_characteristics': self.model.encoder_characteristics.state_dict(),
                'decoder_anatomy': self.model.decoder_anatomy.state_dict(),
                'decoder': self.model.decoder.state_dict(),
            }

            checkpoint_best = {
                'model': self.best_model.state_dict(),
                'encoder_anatomy': self.best_model.encoder_anatomy.state_dict(),
                'encoder_characteristics': self.best_model.encoder_characteristics.state_dict(),
                'decoder_anatomy': self.best_model.decoder_anatomy.state_dict(),
                'decoder': self.best_model.decoder.state_dict(),
            }

        else:
            checkpoint = {
                'model': self.model.state_dict(),
                'encoder': self.model.encoder.state_dict(),
                'decoder': self.model.decoder.state_dict(),
                'decoder_anatomy': self.model.decoder_anatomy.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }

            checkpoint_final = {
                'model': self.model.state_dict(),
                'encoder': self.model.encoder.state_dict(),
                'decoder': self.model.decoder.state_dict(),
                'decoder_anatomy': self.model.decoder_anatomy.state_dict(),
            }

            checkpoint_best = {
                'model': self.best_model.state_dict(),
                'encoder': self.model.encoder.state_dict(),
                'decoder': self.model.decoder.state_dict(),
                'decoder_anatomy': self.model.decoder_anatomy.state_dict(),
            }

        # Save the current checkpoint
        if epoch % save_idx == 0:
            torch.save(checkpoint, self.output_path / f"checkpoint_ep{epoch}.pt")

        # Save the best and final checkpoint
        if epoch == self.epochs:
            torch.save(checkpoint_best, self.output_path / "checkpoint_best.pt")
            torch.save(checkpoint_final, self.output_path / "checkpoint_final.pt")

    def calculate_passed_time(self, start_time, end_time):
        """
        Calculate the time needed for running the code

        :param: start_time: Start time.
        :param: end_time: End time.
        :return: Duration in hh:mm:ss.ss
        """

        # Calculate the duration
        elapsed_time = end_time - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        # Return the duration in hours, minutes and seconds
        return int(hours), int(minutes), seconds


def main(training_configuration=None):
    """
    Initialize the trainer and execute either a training or a hyperparameter optimization run with the provided
    configuration.

    - If a training run is executed the parameter 'training_configuration' contains the respective training parameters.
    - If a hyperparameter optimization run is executed the parameter 'training_configuration' is not used.

    :param training_configuration: Dictionary containing the parameters and hyperparameters of the training run.
    """

    # Initialize either the given training or the hyperparameter optimization weights & biases project
    if training_configuration is not None:
        if training_configuration["resume_training"]["resume"]:
            # Resume the weights & biases project for the specified training run
            wandb.init(project="unoranic+_training", id=training_configuration["resume_training"]["wandb_id"], resume="allow", config=training_configuration)

        else:
            # Initialize a weights & biases project for a training run with the given training configuration
            wandb.init(project="unoranic+_training", name=f"{training_configuration['dataset']}-{training_configuration['architecture']}", config=training_configuration)

    # Run the hyperparameter optimization run
    else:
        # Initialize a weights & biases project for a hyperparameter optimization run
        wandb.init(project="unoranic+_training_sweep")

        # Load the beforehand configured sweep configuration
        training_configuration = wandb.config

    # Initialize the Trainer
    print("Initializing the trainer...")
    trainer = Trainer(training_configuration)

    # Run the training
    print("Train the model...")
    trainer.train()


if __name__ == '__main__':
    # Start code
    start_time = time.time()

    # Read out the command line parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str, help="Path to the configuration file to use.")  # Only for bash execution
    parser.add_argument("--architecture", required=False, default='unoranic+', type=str, help="Which model to use.")
    parser.add_argument("--sweep", required=False, default=False, type=bool, help="Whether to run hyperparameter tuning or just training.")

    args = parser.parse_args()
    config_file = args.config_file
    architecture = args.architecture
    sweep = args.sweep

    # Load the default and the hyperparameters and run the training
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        config['training']['architecture'] = architecture

        if sweep:
            # Configure the sweep
            sweep_id = wandb.sweep(sweep=config['hyperparameter_tuning'], project="unoranic+_training_sweep")

            # Start the sweep
            wandb.agent(sweep_id, function=main, count=100)

        else:
            # If no hyperparameter optimization shall be performed run the training
            main(config['training'])

    # Finished code
    end_time = time.time()

    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))