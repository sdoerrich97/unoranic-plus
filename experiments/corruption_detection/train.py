"""
xAILAB Bamberg
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Training of unORANIC+ for a corruption detection objective.

@author: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
"""

# Import packages
import argparse
from pathlib import Path
import numpy as np
import time
import wandb
import torch
import torch.nn as nn
from copy import deepcopy
import torch.optim as optim
from timm.optim import optim_factory
import yaml
from tqdm import tqdm

# Import own scripts
from data_loader import DataLoaderCustom
from ...models.detector import CorruptionDetector
from ...utils import LearningRate, Metrics


class Trainer:
    def __init__(self, configuration):
        """
        Initialize the trainer.
        """

        # Read out the parameters of the training run
        self.dataset = configuration['dataset']
        self.data_path = Path(configuration['data_path'])
        self.input_path = Path(configuration['input_path'])
        self.output_path = Path(configuration['output_path'])
        self.architecture_encoder = configuration['architecture_encoder']
        self.architecture_decoder = configuration['architecture_decoder']
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
        self.dropout = configuration['dropout']

        self.learning_rate = configuration['optimizer']['learning_rate'] * self.batch_size / 256
        self.learning_rate_min = configuration['optimizer']['learning_rate_min']
        self.first_momentum = configuration['optimizer']['first_momentum']
        self.second_momentum = configuration['optimizer']['second_momentum']
        self.weight_decay = configuration['optimizer']['weight_decay']
        self.warmup_epochs = configuration['optimizer']['warmup_epochs']

        self.task = configuration['task']
        self.nr_classes = configuration['nr_classes']

        # Create the path to where the trained classifier shall be stored
        self.checkpoint_file = self.input_path / f"{self.latent_dim}" / self.dataset / self.architecture_encoder / f"checkpoint_best.pt"

        if self.architecture_encoder == 'resnet18' or self.architecture_encoder == 'resnet50' or self.architecture_encoder == 'ViT':
            self.output_path = self.output_path / f"{self.latent_dim}" / self.dataset / self.architecture_encoder

        else:
            self.output_path = self.output_path / f"{self.latent_dim}" / self.dataset / self.architecture_encoder / self.architecture_decoder

        self.output_path.mkdir(parents=True, exist_ok=True)

        # Enable the use of cudnn
        torch.backends.cudnn.benchmark = True

        # Initialize the dataloader
        print("\tInitializing the dataloader...")
        self.data_loader = DataLoaderCustom(self.dataset, self.data_path, self.img_size, self.batch_size, self.seed,
                                            train=True)
        self.train_loader = self.data_loader.get_train_loader()
        self.val_loader = self.data_loader.get_val_loader()

        # Initialize the model
        print("\tInitializing the model ...")
        if self.architecture_encoder == 'resnet18' or self.architecture_encoder == 'resnet50':
            self.model = CorruptionDetector(self.img_size, self.in_channel, self.latent_dim, self.architecture_encoder,
                                            self.architecture_decoder, self.task, self.nr_classes)

        elif self.architecture_encoder == 'unoranic':
            # Change the hyperparamters for the unoranic and unoranic_adapted models
            self.latent_dim = 256
            self.epochs = 250
            self.optimizer = 'adam'
            self.learning_rate = 0.0001
            self.first_momentum = 0.95
            self.second_momentum = 0.999
            self.scheduler_lr_base = 0.0001
            self.scheduler_lr_max = 0.01
            self.scheduler_step_up = 2000

            self.model = CorruptionDetector(self.img_size, self.in_channel, self.latent_dim, self.architecture_encoder,
                                            self.architecture_decoder, self.task, self.nr_classes, dropout=self.dropout)

        elif self.architecture_encoder == 'ViT' or self.architecture_encoder == 'unoranic+':
            self.model = CorruptionDetector(self.img_size, self.in_channel, self.latent_dim, self.architecture_encoder,
                                            self.architecture_decoder, self.task, self.nr_classes, patch_size=self.patch_size,
                                            depth=self.depth, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
                                            norm_layer=self.norm_layer)

        else:
            raise SystemExit

        # Load the trained encoder with reconstruction objective
        if self.architecture_encoder == 'unoranic':
            checkpoint = torch.load(self.checkpoint_file, map_location='cpu')
            self.model.encoder_anatomy.load_state_dict(checkpoint['encoder_anatomy'])
            self.model.encoder_characteristics.load_state_dict(checkpoint['encoder_characteristics'])

        elif self.architecture_encoder == 'unoranic+':
            checkpoint = torch.load(self.checkpoint_file, map_location='cpu')
            self.model.encoder.load_state_dict(checkpoint['encoder'])

            # For the ViT decoder, load the pretrained decoder as well but adapt for the classification task
            if self.architecture_decoder == 'ViT':
                del checkpoint['decoder']['pos_embed']
                del checkpoint['decoder']['pred.weight']
                del checkpoint['decoder']['pred.bias']

                self.model.decoder.load_state_dict(checkpoint['decoder'], strict=False)

        # Map the model to the GPU and enable training for the required parts
        self.model = self.model.to(self.device)
        self.model.requires_grad_(True)

        if self.architecture_encoder == 'resnet18' or self.architecture_encoder == 'resnet50' or self.architecture_encoder == 'ViT':
            self.model.requires_grad_(True)

        else:
            self.model.encoder.requires_grad_(False)

        # Initialize the optimizer and learning rate scheduler
        print("\tInitializing the optimizer and lr scheduler...")
        if self.architecture_encoder == 'resnet18' or self.architecture_encoder == 'resnet50':
            lr = 0.001
            gamma = 0.1
            milestones = [0.5 * self.epochs, 0.75 * self.epochs]

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)

        elif self.architecture_encoder == 'unoranic':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                              betas=(self.first_momentum,
                                                     self.second_momentum))
            self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.scheduler_lr_base,
                                                         max_lr=self.scheduler_lr_max,
                                                         step_size_up=len(self.train_loader) * 5,
                                                         mode='triangular2', cycle_momentum=False)

        else:
            param_groups = optim_factory.param_groups_weight_decay(self.model, self.weight_decay)
            self.optimizer = torch.optim.AdamW(param_groups, lr=self.learning_rate, betas=(self.first_momentum,
                                                                                           self.second_momentum))

        # Initialize the loss criterion and variables to store the best performing model
        print("\tInitialize the loss criterion and helper variables...")
        self.best_loss = np.inf
        self.best_epoch = self.starting_epoch
        self.best_model = deepcopy(self.model)
        self.loss_criterion = nn.CrossEntropyLoss().to(self.device)

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
        Run one training/validation iteration of the classifier.

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

        # Initialize the loss value
        loss = 0
        Y_target, Y_predicted = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)

        # Iterate through all samples
        for i, (_, _, X, Y, _) in enumerate(data_loader):
            # Map the input to the respective device
            X, Y = X.to(self.device), Y.to(self.device).reshape(-1)

            # Run the input through the model
            Z = self.model(X)

            # Calculate the loss
            loss_current = self.loss_criterion(Z, Y)

            # Run the backward pass for train mode
            if mode == 'Train':
                # Zero out the gradients
                self.optimizer.zero_grad()

                # Run the backward pass for the MSE loss
                loss_current.backward()

                # Optimize the weights
                self.optimizer.step()

                # Adjust the learning rate
                if not (self.architecture_encoder == 'resnet18' or self.architecture_encoder == 'resnet50' or self.architecture_encoder == 'unoranic'):
                    LearningRate.adjust_learning_rate(self.optimizer, i / len(data_loader) + epoch, self.learning_rate,
                                                      self.learning_rate_min, self.epochs, self.warmup_epochs)

            # Append the current loss value to the overall value for the current epoch and store the true and predicted
            # labels
            loss += loss_current.detach().cpu()
            Y_target = torch.cat((Y_target, Y.detach()), dim=0)
            Y_predicted = torch.cat((Y_predicted, Z.detach()), dim=0)

            # Update the progress bar
            pbar.update(1)

        # Update the learning rate scheduler for the baseline resnet architectures
        if mode == 'Train' and (self.architecture_encoder == 'resnet18' or self.architecture_encoder == 'resnet50' or self.architecture_encoder == 'unoranic'):
            self.scheduler.step()

        # Average the loss value across all batches and compute the performance metrics
        loss /= len(data_loader)

        ACC = Metrics.getACC(Y_target.cpu().numpy(), Y_predicted.cpu().numpy(), self.task)
        AUC = Metrics.getAUC(Y_target.cpu().numpy(), Y_predicted.cpu().numpy(), self.task)

        # Print the loss values and send them to wandb
        print(f"\t\t\t{mode} Cross-Entropy (CE) Loss: {loss}")
        print(f"\t\t\t{mode} Accuracy (ACC): {ACC}")
        print(f"\t\t\t{mode} Area-Under-The-Curve (AUC): {AUC}")

        wandb.log(
            {
                f"{mode} Cross-Entropy (CE) Loss": loss,
                f"{mode} Accuracy (ACC)": ACC,
                f"{mode} Area-Under-The-Curve (AUC)": AUC,
            }, step=epoch)

        # Store the current best model
        if mode == 'Val':
            if loss < self.best_loss:
                self.best_epoch = epoch
                self.best_loss = loss
                self.best_model = deepcopy(self.model)

            print(f"\t\t\tCurrent best Cross-Entropy (CE) Loss: {self.best_loss}")
            print(f"\t\t\tCurrent best Epoch: {self.best_epoch}")

    def save_model(self, epoch: int, save_idx=50):
        """
        Save the model every save_idx epochs.

        :param epoch: Current epoch.
        :param save_idx: Which epochs to save.
        """

        # Create the checkpoint
        if self.architecture_encoder == 'unoranic':
            checkpoint = {
                'model': self.model.state_dict(),
                'encoder_anatomy': self.model.encoder_anatomy.state_dict(),
                'encoder_characteristics': self.model.encoder_characteristics.state_dict(),
                'decoder': self.model.decoder.state_dict(),
                'predictor': self.model.predictor.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()
            }

            checkpoint_final = {
                'model': self.model.state_dict(),
                'encoder_anatomy': self.model.encoder_anatomy.state_dict(),
                'encoder_characteristics': self.model.encoder_characteristics.state_dict(),
                'decoder': self.model.decoder.state_dict(),
                'predictor': self.model.predictor.state_dict(),
            }

            checkpoint_best = {
                'model': self.best_model.state_dict(),
                'encoder_anatomy': self.best_model.encoder_anatomy.state_dict(),
                'encoder_characteristics': self.best_model.encoder_characteristics.state_dict(),
                'decoder': self.best_model.decoder.state_dict(),
                'predictor': self.best_model.predictor.state_dict(),
            }
        else:
            if self.architecture_encoder == 'ViT' or self.architecture_encoder == 'unoranic+':
                checkpoint = {
                    'model': self.model.state_dict(),
                    'encoder': self.model.encoder.state_dict(),
                    'decoder': self.model.decoder.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }
            else:
                checkpoint = {
                    'model': self.model.state_dict(),
                    'encoder': self.model.encoder.state_dict(),
                    'decoder': self.model.decoder.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                }

            checkpoint_final = {
                'model': self.model.state_dict(),
                'encoder': self.model.encoder.state_dict(),
                'decoder': self.model.decoder.state_dict(),
                'predictor': self.model.predictor.state_dict(),
            }

            checkpoint_best = {
                'model': self.best_model.state_dict(),
                'encoder': self.best_model.encoder.state_dict(),
                'decoder': self.best_model.decoder.state_dict(),
                'predictor': self.best_model.predictor.state_dict(),
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
            wandb.init(project="unoranic+_corruption_detection_training", id=training_configuration["resume_training"]["wandb_id"], resume="allow", config=training_configuration)

        else:
            # Initialize a weights & biases project for a training run with the given training configuration
            wandb.init(project="unoranic+_corruption_detection_training", name=f"{training_configuration['dataset']}-{training_configuration['architecture_encoder']}", config=training_configuration)

    # Run the hyperparameter optimization run
    else:
        # Initialize a weights & biases project for a hyperparameter optimization run
        wandb.init(project="unoranic+_corruption_detection_training_sweep")

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
    parser.add_argument("--architecture_encoder", required=False, default='unoranic+', type=str, help="Which model to use for the encoder.")
    parser.add_argument("--architecture_decoder", required=False, default='single_layer', type=str, help="Which model to use for the decoder.")
    parser.add_argument("--sweep", required=False, default=False, type=bool, help="Whether to run hyperparameter tuning or just training.")

    args = parser.parse_args()
    config_file = args.config_file
    architecture_encoder = args.architecture_encoder
    architecture_decoder = args.architecture_decoder
    sweep = args.sweep

    # Load the default and the hyperparameters and run the training
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        config['training']['architecture_encoder'] = architecture_encoder
        config['training']['architecture_decoder'] = architecture_decoder

        if sweep:
            # Configure the sweep
            sweep_id = wandb.sweep(sweep=config['hyperparameter_tuning'], project="unoranic+_corruption_detection_training_sweep")

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