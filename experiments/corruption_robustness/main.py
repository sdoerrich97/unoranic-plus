"""
xAILAB Bamberg
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Main script of unORANIC+ for a corruption robustness objective.

@author: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
"""

# Import packages
import argparse
from pathlib import Path
import time
import wandb
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
from imagecorruptions import get_corruption_names
import json

# Import own scripts
from data_loader import DataLoaderCustom
from ...models.classifier import Classifier
from ...utils import Metrics, Plotting


class CorruptionRobustness:
    def __init__(self, configuration):
        """
        Initialize the corruption robustness.
        """

        # Read out the parameters of the training run
        self.dataset = configuration['dataset']
        self.data_path = Path(configuration['data_path'])
        self.input_path = Path(configuration['input_path'])
        self.output_path = Path(configuration['output_path'])
        self.architecture_encoder = configuration['architecture_encoder']
        self.architecture_decoder = configuration['architecture_decoder']
        self.seed = configuration['seed']
        self.device = torch.device(configuration['device'])

        # Read out the hyperparameters of the training run
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

        self.task = configuration['task']
        self.nr_classes = configuration['nr_classes']

        # Create the path to where the trained classifier is stored and the output shall be stored
        if self.architecture_encoder == 'resnet18' or self.architecture_encoder == 'resnet50' or self.architecture_encoder == 'ViT':
            self.checkpoint_file = self.input_path / f"{self.latent_dim}" / self.dataset / self.architecture_encoder / "checkpoint_best.pt"
            self.output_path = self.output_path / f"{self.latent_dim}" / self.dataset / self.architecture_encoder

        else:
            self.checkpoint_file = self.input_path / f"{self.latent_dim}" / self.dataset / self.architecture_encoder / self.architecture_decoder / "checkpoint_best.pt"
            self.output_path = self.output_path / f"{self.latent_dim}" / self.dataset / self.architecture_encoder / self.architecture_decoder

        self.output_path.mkdir(parents=True, exist_ok=True)

        # Enable the use of cudnn
        torch.backends.cudnn.benchmark = True

        # Initialize the model
        print("\tInitializing the model ...")
        self.checkpoint = torch.load(self.checkpoint_file, map_location='cpu')

        if self.architecture_encoder == 'resnet18' or self.architecture_encoder == 'resnet50':
            self.model = Classifier(self.img_size, self.in_channel, self.latent_dim, self.architecture_encoder,
                                    self.architecture_decoder, self.task, self.nr_classes)

        elif self.architecture_encoder == 'unoranic':
            self.latent_dim = 256  # Change the hyperparamters for the unoranic and unoranic_adapted models
            self.model = Classifier(self.img_size, self.in_channel, self.latent_dim, self.architecture_encoder,
                                    self.architecture_decoder, self.task, self.nr_classes, dropout=self.dropout)

        elif self.architecture_encoder == 'ViT' or self.architecture_encoder == 'unoranic+':
            self.model = Classifier(self.img_size, self.in_channel, self.latent_dim, self.architecture_encoder,
                                    self.architecture_decoder, self.task, self.nr_classes, patch_size=self.patch_size,
                                    depth=self.depth, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
                                    norm_layer=self.norm_layer)

        else:
            raise SystemExit

        # Load the trained weights
        self.model.load_state_dict(self.checkpoint['model'])
        self.model = self.model.to(self.device)
        self.model.requires_grad_(False)

        # Initialize the loss criterion
        print("\tInitialize the loss criterion...")
        if self.task == "multi-label, binary-class":
            self.loss_criterion = nn.BCEWithLogitsLoss().to(self.device)
        else:
            self.loss_criterion = nn.CrossEntropyLoss().to(self.device)

    def run_corruption_robustness(self):
        """
        Run the corruption robustness experiment.
        """

        # Start code
        start_time = time.time()

        # Empty the unused memory cache
        print("\tRun corruption robustness...")
        print("\tEmptying the unused memory cache...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Create dictionaries to store the corrupted images and the respective metrics
        step = 0
        X_unbiased, all_corrupted_variants, all_auc, all_acc = {}, {}, {}, {}

        # Iterate through all corruptions and test the robustness of the model
        for corruption in get_corruption_names():
            # Skip the broken corruptions
            if corruption == 'glass_blur':
                continue

            # Run the corruption robustness for the current corruption
            start_time_corruption = time.time()
            print(f"\t\tRun corruption robustness for {corruption}...")

            # Iterate through all severity degrees and test the robustness of the model
            for severity in range(1, 6):
                # Run the corruption robustness for the current corruption - severity combination
                start_time_severity = time.time()
                print(f"\t\t\tRun corruption robustness for severity {severity}...")

                # Create the data loader for the current corruption
                data_loader = DataLoaderCustom(self.dataset, self.data_path, self.img_size, self.in_channel, self.batch_size, corruption, severity)
                train_loader = data_loader.get_train_loader()
                val_loader = data_loader.get_val_loader()
                test_loader = data_loader.get_test_loader()

                for mode, set_loader in zip(['Train', 'Val', 'Test'], [train_loader, val_loader, test_loader]):
                    # Run the corruption robustness for the current mode
                    start_time_mode = time.time()
                    print(f"\t\t\t\tRun corruption robustness for {mode}...")

                    nr_batches = len(set_loader)
                    print(f"\t\t\t\t\tRun corruption robustness for {nr_batches} batches...")

                    with torch.no_grad():
                        # Set the model into evaluation mode
                        self.model.eval()

                        # Run the corruption robustness for the current corruption-severity combination on the current mode
                        X_orig, X, ACC, AUC = self.run_corruption_robustness_for_single_mode(mode, set_loader, step, nr_batches)

                        # Store the corrupted image and the metrics
                        if severity == 1 and mode == 'Train':
                            X_unbiased = {mode: X_orig}
                            all_corrupted_variants[corruption] = {mode: [X]}
                            all_acc[corruption] = {mode: [ACC]}
                            all_auc[corruption] = {mode: [AUC]}

                        elif severity == 1:
                            X_unbiased[mode] = X_orig
                            all_corrupted_variants[corruption][mode] = [X]
                            all_acc[corruption][mode] = [ACC]
                            all_auc[corruption][mode] = [AUC]

                        else:
                            all_corrupted_variants[corruption][mode].append(X)
                            all_acc[corruption][mode].append(ACC)
                            all_auc[corruption][mode].append(AUC)

                    # Stop the time
                    end_time_mode = time.time()
                    hours_mode, minutes_mode, seconds_mode = self.calculate_passed_time(start_time_mode, end_time_mode)

                    print("\t\t\t\t\tElapsed time for mode '{}': {:0>2}:{:0>2}:{:05.2f}".format(mode, hours_mode, minutes_mode, seconds_mode))

                # Increment the step
                step += 1

                # Stop the time
                end_time_severity = time.time()
                hours_severity, minutes_severity, seconds_severity = \
                    self.calculate_passed_time(start_time_severity, end_time_severity)

                print("\t\t\t\tElapsed time for severity '{}': {:0>2}:{:0>2}:{:05.2f}".format(severity,
                                                                                              hours_severity,
                                                                                              minutes_severity,
                                                                                              seconds_severity))
            # Stop the time
            end_time_corruption = time.time()
            hours_corruption, minutes_corruption, seconds_corruption = \
                self.calculate_passed_time(start_time_corruption, end_time_corruption)

            print("\t\t\tElapsed time for corruption '{}': {:0>2}:{:0>2}:{:05.2f}".format(corruption,
                                                                                          hours_corruption,
                                                                                          minutes_corruption,
                                                                                          seconds_corruption))

        # Create an evaluation file for the metrics for each set
        self.create_evaluations(all_acc, all_auc, self.output_path)

        # Plot one reference image together with its corruptions for each set
        self.plot_original_image_and_corrupted_variants(X_unbiased, all_corrupted_variants, self.output_path)

        # Stop the time
        end_time = time.time()
        hours, minutes, seconds = self.calculate_passed_time(start_time, end_time)

        print("\t\t\tElapsed time for corruption robustness: {:0>2}:{:0>2}:{:05.2f}".format(hours, minutes, seconds))

    def run_corruption_robustness_for_single_mode(self, mode, set_loader, step, nr_batches):
        """
        Run the corruption robustness for the classifier model.

        :param mode: Train, Val or Test set.
        :param set_loader: Data loader for the current set.
        :param step: Index for wandb
        :param nr_batches: Number of batches to process.
        """

        # Initialize a progress bar
        pbar = tqdm(total=len(set_loader), bar_format='{l_bar}{bar}', ncols=80, initial=0, position=0, leave=False)
        pbar.set_description(f'\t\t\t\t\tProgres')

        loss = 0
        X_orig_0, X_0 = None, None
        Y_target, Y_predicted = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)

        for i, (X_orig, Y, X, _) in enumerate(set_loader):
            # Map the input to the respective device
            if self.task == "multi-label, binary-class":
                X, Y = X.to(self.device), Y.to(torch.float32).to(self.device)
            else:
                X, Y = X.to(self.device), Y.to(self.device).reshape(-1)

            # Run the input through the model
            Z = self.model(X)

            # Calculate the loss
            loss_batch = self.loss_criterion(Z, Y)

            print(f"\t\t\t\t\t\tBatch: [{i}/{nr_batches}]")
            print(f"\t\t\t\t\t\tCrossEntropyLoss: {loss_batch}")

            # Store the target and predicted labels
            loss += loss_batch.detach().cpu()
            Y_target = torch.cat((Y_target, Y.detach()), dim=0)
            Y_predicted = torch.cat((Y_predicted, Z.detach()), dim=0)

            # Store the first image
            if i == 0:
                X_orig_0, X_0 = X_orig[0].cpu(), X[0].cpu()

            # Update the progress bar
            pbar.update(1)

        # Average the loss value across all batches and compute the performance metrics
        loss /= nr_batches

        ACC = Metrics.getACC(Y_target.cpu().numpy(), Y_predicted.cpu().numpy(), self.task)
        AUC = Metrics.getAUC(Y_target.cpu().numpy(), Y_predicted.cpu().numpy(), self.task)

        # Print the loss values and send them to wandb
        print(f"\t\t\t{mode} Average Cross-Entropy (CE) Loss: {loss}")
        print(f"\t\t\t{mode} Average Accuracy (ACC): {ACC}")
        print(f"\t\t\t{mode} Average Area-Under-The-Curve (AUC): {AUC}")

        # Log the metrics' averages
        wandb.log(
            {
                f"{mode} Average Cross-Entropy (CE) Loss": loss,
                f"{mode} Average Accuracy (ACC)": ACC,
                f"{mode} Average Area-Under-The-Curve (AUC)": AUC
            }, step=step)

        # Return the very first image and its corruption as well as the metric values
        return X_orig_0, X_0, ACC, AUC

    def create_evaluations(self, ACC: dict, AUC: dict, output_path: Path):
        """
        Save accuracy and AUC values for each set in a unique evaluation file.

        :param ACC: Accuracy values for each set across all corruption-severity pairs.
        :param AUC: AUC values for each set across all corruption-severity pairs.
        :param output_path: Where the evaluation shall be stored.
        """

        # Initialize the evaluation as a dictionary
        evaluation = {}

        # Create one evaluation for each set
        for mode in ["Train", "Val", "Test"]:
            # Create an entry for the current set
            evaluation[mode] = {}

            # Iterate over all corruptions and extract the metric values for all severities
            for corruption in ACC.keys():
                evaluation[mode][corruption] = {'ACC': ACC[corruption][mode], 'AUC': AUC[corruption][mode]}

            # Write the evlaluation for each set in a new JSON file
            with open((output_path / f"evaluation_{mode}.json"), 'w') as json_file:
                json.dump(evaluation[mode], json_file, indent=4)

    def plot_original_image_and_corrupted_variants(self, X_orig: dict, images: dict, output_path: Path):
        """
        Save accuracy and AUC values for each set in a unique evaluation file.

        :param X_orig: Original, uncorrupted images for each set.
        :param images: Dictionary containing the corrupted variants for each corruption-severity combination.
        :param output_path: Where the images shall be stored
        """

        # Create one plot for each set and severity across all corruptions
        for mode in ["Train", "Val", "Test"]:
            for severity in range(1, 6):
                # Create a list for all corrupted variants
                corrupted_variants = []

                # Iterate over all corruptions and extract the metric values for all severities
                for corruption in images.keys():
                    corrupted_variants.append(images[corruption][mode][severity - 1])

                # Plot the corruptions individually and in a single plot for the current set
                Plotting.plot_image_and_corrupted_variants(mode, list(images.keys()), severity, X_orig[mode],
                                                           corrupted_variants, self.in_channel, output_path)

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


def main(configuration):
    """
    Run the corruption robustness.

    :param configuration: Configuration of the corruption robustness run.
    """

    # Initialize the weights & biases project with the given corruption robustness configuration
    wandb.init(project="unoranic+_corruption_robustness", name=f"{configuration['dataset']}-{configuration['architecture_encoder']}", config=configuration)

    # Initialize the corruption robustness
    print("Initializing the corruption robustness ...")
    cr = CorruptionRobustness(configuration)

    # Run the corruption robustness
    print("Run the corruption robustness...")
    cr.run_corruption_robustness()


if __name__ == '__main__':
    # Start code
    start_time = time.time()

    # Read out the command line parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str, help="Path to the configuration file to use.")
    parser.add_argument("--architecture_encoder", required=False, default='unoranic+', type=str, help="Which model to use for the encoder.")
    parser.add_argument("--architecture_decoder", required=False, default='single_layer', type=str, help="Which model to use for the decoder.")
    #parser.add_argument("--latent_dim", required=False, default=128, type=int, help="Which model to use.")

    args = parser.parse_args()
    config_file = args.config_file
    architecture_encoder = args.architecture_encoder
    architecture_decoder = args.architecture_decoder
    #latent_dim = args.latent_dim

    # Load the default and the hyperparameters and run the training
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        config['architecture_encoder'] = architecture_encoder
        config['architecture_decoder'] = architecture_decoder
        #config['latent_dim'] = latent_dim

        main(config)

    # Finished code
    end_time = time.time()

    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))