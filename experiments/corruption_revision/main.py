"""
xAILAB Bamberg
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Main script of unORANIC+ for a corruption revision objective.

@author: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
"""

# Import packages
import argparse
from pathlib import Path
import time
import wandb
import torch
import yaml
from tqdm import tqdm

# Import own scripts
from data_loader import DataLoaderCustom
from ...models.unoranic import Unoranic
from ...models.unoranic_plus import UnoranicPlus
from ...utils import Plotting


class CorruptionRevision:
    def __init__(self, configuration):
        """
        Initialize the corruption revision.
        """

        # Read out the parameters of the training run
        self.dataset = configuration['dataset']
        self.data_path = Path(configuration['data_path'])
        self.input_path = Path(configuration['input_path'])
        self.output_path = Path(configuration['output_path'])
        self.architecture = configuration['architecture']
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

        # Create the path to where the output shall be stored and initialize the logger
        self.output_path = self.output_path / f"{self.latent_dim}" / self.dataset / self.architecture
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Enable the use of cudnn
        torch.backends.cudnn.benchmark = True

        # Initialize the model
        print("\tInitializing the model ...")
        self.checkpoint_file = self.input_path / f"{self.latent_dim}" / self.dataset / self.architecture / "checkpoint_best.pt"
        self.checkpoint = torch.load(self.checkpoint_file, map_location='cpu')

        if self.architecture == 'unoranic':
            self.latent_dim = 256  # Change the hyperparamters for the convolutional autoencoder
            self.model = Unoranic(self.img_size, self.in_channel, self.latent_dim, self.dropout)

        elif self.architecture == 'unoranic+':
            self.model = UnoranicPlus(self.img_size, self.in_channel, self.patch_size, self.latent_dim, self.depth,
                                      self.num_heads, self.mlp_ratio, self.norm_layer)

        else:
            raise SystemExit

        # Load the trained weights
        self.model.load_state_dict(self.checkpoint['model'])
        self.model.requires_grad_(False)

        # Load the corruptions of the Albumentations Library
        self.corruptions = ['PixelDropout', 'GaussianBlur', 'ColorJitter', 'Downscale', 'GaussNoise', 'InvertImg',
                            'MotionBlur', 'MultiplicativeNoise', 'RandomBrightnessContrast', 'RandomGamma', 'Solarize',
                            'Sharpen']

    def run_corruption_revision(self):
        """
        Run the corruption revision experiment.
        """

        # Start code
        start_time = time.time()

        # Empty the unused memory cache
        print("\tRun corruption revision...")
        print("\tEmptying the unused memory cache...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for c, corruption in enumerate(self.corruptions):
            # Run the corruption revision for the current corruption
            start_time_corruption = time.time()
            print(f"\t\tRun corruption revision for {corruption}...")

            # Create the output path for the current corruption
            output_path = self.output_path / corruption
            output_path.mkdir(parents=True, exist_ok=True)

            # Create the data loader for the current corruption
            data_loader = DataLoaderCustom(self.dataset, self.data_path, self.img_size, self.batch_size, corruption, self.seed)
            train_loader = data_loader.get_train_loader()
            val_loader = data_loader.get_val_loader()
            test_loader = data_loader.get_test_loader()

            for set, set_loader in zip(['Train', 'Val', 'Test'], [train_loader, val_loader, test_loader]):
                # Run the corruption revision for the current set
                start_time_set = time.time()
                print(f"\t\t\tRun corruption revision for {set}...")

                nr_batches = len(set_loader)
                print(f"\t\t\t\tRun corruption revision for {nr_batches} batches...")

                with torch.no_grad():
                    # Set the model into evaluation mode
                    self.model.eval()

                    # Run the corruption revision
                    self.run_corruption_revision_for_single_set(set, set_loader, c, nr_batches, output_path)

                # Stop the time
                end_time_set = time.time()
                hours_set, minutes_set, seconds_set = self.calculate_passed_time(start_time_set, end_time_set)

                print("\t\t\t\tElapsed time for set '{}': {:0>2}:{:0>2}:{:05.2f}".format(set, hours_set, minutes_set, seconds_set))

            # Stop the time
            end_time_corruption = time.time()
            hours_corruption, minutes_corruption, seconds_corruption = \
                self.calculate_passed_time(start_time_corruption, end_time_corruption)

            print("\t\t\tElapsed time for corruption '{}': {:0>2}:{:0>2}:{:05.2f}".format(corruption,
                                                                                          hours_corruption,
                                                                                          minutes_corruption,
                                                                                          seconds_corruption))

        # Stop the time
        end_time = time.time()
        hours, minutes, seconds = self.calculate_passed_time(start_time, end_time)

        print("\t\t\tElapsed time for corruption revision: {:0>2}:{:0>2}:{:05.2f}".format(hours, minutes, seconds))

    def run_corruption_revision_for_single_set(self, mode, set_loader, c, nr_batches, output_path):
        """
        Run the corruption revision for the given set.

        :param mode: Train, Val or Test set.
        :param set_loader: Data loader for the current set.
        :param c: Corruption index.
        :param nr_batches: Number of batches to process.
        :param output_path: Where the image shall be stored.
        """

        # Initialize a progress bar
        pbar = tqdm(total=len(set_loader), bar_format='{l_bar}{bar}', ncols=80, initial=0, position=0, leave=False)
        pbar.set_description(f'\t\t\tProgres')

        mse_input, mse_anatomy, mse_characteristics = 0, 0, 0
        ssim_input, ssim_anatomy, ssim_characteristics = 0, 0, 0
        psnr_input, psnr_anatomy, psnr_characteristics = 0, 0, 0

        for i, (X_orig, _, X, X_name) in enumerate(set_loader):
            # Run the input through the model
            if self.architecture == 'unoranic':
                (mse_input_batch, mse_anatomy_batch, mse_characteristics_batch), \
                (ssim_input_batch, ssim_anatomy_batch, ssim_characteristics_batch), \
                (psnr_input_batch, psnr_anatomy_batch, psnr_characteristics_batch), \
                _, _, (X_diff, X_hat, X_hat_anatomy, X_hat_diff) = self.model(X_orig, X, [], [])

            else:
                (mse_input_batch, mse_anatomy_batch, mse_characteristics_batch), \
                (ssim_input_batch, ssim_anatomy_batch, ssim_characteristics_batch), \
                (psnr_input_batch, psnr_anatomy_batch, psnr_characteristics_batch), \
                _, _, (X_diff, X_hat, X_hat_anatomy, X_hat_diff) = self.model(X_orig, X)

            # Append the current loss values to the overall values for the set
            mse_input += mse_input_batch.detach().cpu()
            mse_anatomy += mse_anatomy_batch.detach().cpu()
            mse_characteristics += mse_characteristics_batch.detach().cpu()
            ssim_input += ssim_input_batch.detach().cpu()
            ssim_anatomy += ssim_anatomy_batch.detach().cpu()
            ssim_characteristics += ssim_characteristics_batch.detach().cpu()
            psnr_input += psnr_input_batch.detach().cpu()
            psnr_anatomy += psnr_anatomy_batch.detach().cpu()
            psnr_characteristics += psnr_characteristics_batch.detach().cpu()

            # Save the very first sample for each set as well as its reconstruction
            if i == 0:
                Plotting.save_img_and_reconstructed_image(X_orig[0].cpu(), X[0].cpu(), X_diff[0], X_hat[0],
                                                          X_hat_anatomy[0], X_hat_diff[0], f"{mode}_{X_name[0][0]}",
                                                          output_path)

            # Update the progress bar
            pbar.update(1)

        # Average the metrics over the number of samples for the current set
        mse_input /= nr_batches
        mse_anatomy /= nr_batches
        mse_characteristics /= nr_batches
        ssim_input /= nr_batches
        ssim_anatomy /= nr_batches
        ssim_characteristics /= nr_batches
        psnr_input /= nr_batches
        psnr_anatomy /= nr_batches
        psnr_characteristics /= nr_batches

        # Print the loss values and send them to wandb
        print(f"\t\t\t\t{mode} Average MSE Input: {mse_input}")
        print(f"\t\t\t\t{mode} Average MSE Anatomy: {mse_anatomy}")
        print(f"\t\t\t\t{mode} Average MSE Characteristics: {mse_characteristics}")
        print(f"\t\t\t\t{mode} Average SSIM Input: {ssim_input}")
        print(f"\t\t\t\t{mode} Average SSIM Anatomy: {ssim_anatomy}")
        print(f"\t\t\t\t{mode} Average SSIM Characteristics: {ssim_characteristics}")
        print(f"\t\t\t\t{mode} Average PSNR Input: {psnr_input}")
        print(f"\t\t\t\t{mode} Average PSNR Anatomy: {psnr_anatomy}")
        print(f"\t\t\t\t{mode} Average PSNR Characteristics: {psnr_characteristics}")

        # Log the metrics' averages
        wandb.log(
            {
                f"{mode} Average MSE Input": mse_input,
                f"{mode} Average MSE Anatomy": mse_anatomy,
                f"{mode} Average MSE Characteristics": mse_characteristics,
                f"{mode} Average SSIM Input": ssim_input,
                f"{mode} Average SSIM Anatomy": ssim_anatomy,
                f"{mode} Average SSIM Characteristics": ssim_characteristics,
                f"{mode} Average PSNR Input": psnr_input,
                f"{mode} Average PSNR Anatomy": psnr_anatomy,
                f"{mode} Average PSNR Characteristics": psnr_characteristics,
            }, step=c)

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
    Run the corruption revision.

    :param configuration: Configuration of the corruption revision run.
    """

    # Initialize the weights & biases project with the given corruption revision configuration
    wandb.init(project="unoranic+_corruption_revision_experiment", name=f"{configuration['dataset']}-{configuration['architecture']}", config=configuration)

    # Initialize the corruption revision
    print("Initializing the corruption revision ...")
    cr = CorruptionRevision(configuration)

    # Run the corruption revision
    print("Run the corruption revision...")
    cr.run_corruption_revision()


if __name__ == '__main__':
    # Start code
    start_time = time.time()

    # Read out the command line parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str, help="Path to the configuration file to use.")
    parser.add_argument("--architecture", required=False, default='unoranic+', type=str, help="Which model to use.")

    args = parser.parse_args()
    config_file = args.config_file
    architecture = args.architecture

    # Load the default and the hyperparameters and run the training
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        config['architecture'] = architecture

        main(config)

    # Finished code
    end_time = time.time()

    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))