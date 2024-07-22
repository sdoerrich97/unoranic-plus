"""
xAILAB Bamberg
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Inference.

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
from ..models.unoranic import Unoranic
from ..models.unoranic_plus import UnoranicPlus
from ..utils import Plotting


class Inference:
    def __init__(self, configuration):
        """
        Initialize the Inferer.
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
        self.tuple_size = configuration['tuple_size']
        self.dropout = configuration['dropout']

        # Create the path to where the output shall be stored and initialize the logger
        self.output_path = self.output_path / f"{self.latent_dim}" / self.dataset / self.architecture
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Enable the use of cudnn
        torch.backends.cudnn.benchmark = True

        # Initialize the dataloader
        print("\tInitializing the dataloader...")
        self.data_loader = DataLoaderCustom(self.dataset, self.data_path, self.img_size, self.batch_size, self.tuple_size,
                                            train=False)
        self.train_loader = self.data_loader.get_train_loader()
        self.val_loader = self.data_loader.get_val_loader()
        self.test_loader = self.data_loader.get_test_loader()

        # Initialize the model
        print("\tInitializing the model ...")
        self.checkpoint_file = self.input_path / f"{self.latent_dim}" / self.dataset / self.architecture / "checkpoint_best.pt"
        self.checkpoint = torch.load(self.checkpoint_file, map_location='cpu')

        if self.architecture == 'unoranic':
            self.latent_dim = 256  # Change the hyperparamters for the unoranic and unoranic_adapted models
            self.model = Unoranic(self.img_size, self.in_channel, self.latent_dim, self.dropout)

        elif self.architecture == 'unoranic+':
            self.model = UnoranicPlus(self.img_size, self.in_channel, self.patch_size, self.latent_dim, self.depth,
                                      self.num_heads, self.mlp_ratio, self.norm_layer)

        else:
            raise SystemExit

        # Load the trained model
        self.model.load_state_dict(self.checkpoint['model'])
        self.model.requires_grad_(False)

    def run_inference(self):
        """
        Run the inference.
        """

        # Start code
        start_time = time.time()

        # Empty the unused memory cache
        print("\tRun inference...")
        print("\tEmptying the unused memory cache...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for set, set_loader in zip(['train', 'val', 'test'], [self.train_loader, self.val_loader, self.test_loader]):
            # Run the inference for the current set
            start_time_set = time.time()
            print(f"\t\tRun inference for {set}...")

            # Create the output path for the current set
            output_path = self.output_path / set
            output_path.mkdir(parents=True, exist_ok=True)

            nr_batches = len(set_loader)
            print(f"\t\t\tRun inference for {nr_batches} batches...")

            with torch.no_grad():
                # Set the model into evaluation mode
                self.model.eval()

                # Run the inference
                self.run_inference_for_single_set(set_loader, nr_batches, output_path)

            # Stop the time
            end_time_set = time.time()
            hours_set, minutes_set, seconds_set = self.calculate_passed_time(start_time_set, end_time_set)

            print("\t\t\tElapsed time for set '{}': {:0>2}:{:0>2}:{:05.2f}".format(set, hours_set, minutes_set, seconds_set))

        # Stop the time
        end_time = time.time()
        hours, minutes, seconds = self.calculate_passed_time(start_time, end_time)

        print("\t\t\tElapsed time for inference: {:0>2}:{:0>2}:{:05.2f}".format(hours, minutes, seconds))

    def run_inference_for_single_set(self, set_loader, nr_batches, output_path):
        """
        Run the inference for the given set.

        :param set_loader: Data loader for the current set.
        :param nr_batches: Number of batches to process.
        :param output_path: Where the image shall be stored.
        """

        # Initialize a progress bar
        pbar = tqdm(total=len(set_loader), bar_format='{l_bar}{bar}', ncols=80, initial=0, position=0, leave=False)
        pbar.set_description(f'\t\t\tProgres')

        mse_input, mse_anatomy, mse_characteristics = 0, 0, 0
        ssim_input, ssim_anatomy, ssim_characteristics = 0, 0, 0
        psnr_input, psnr_anatomy, psnr_characteristics = 0, 0, 0

        for i, (X_orig, _, X, _, _, X_name) in enumerate(set_loader):
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
                Plotting.save_img_and_reconstructed_image(X_orig[0].cpu(), X[0].cpu(), X_diff[0], X_hat[0], X_hat_anatomy[0],
                                                          X_hat_diff[0], X_name[0], output_path)

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
        print(f"\t\t\tAverage MSE Input: {mse_input}")
        print(f"\t\t\tAverage MSE Anatomy: {mse_anatomy}")
        print(f"\t\t\tAverage MSE Characteristics: {mse_characteristics}")
        print(f"\t\t\tAverage SSIM Input: {ssim_input}")
        print(f"\t\t\tAverage SSIM Anatomy: {ssim_anatomy}")
        print(f"\t\t\tAverage SSIM Characteristics: {ssim_characteristics}")
        print(f"\t\t\tAverage PSNR Input: {psnr_input}")
        print(f"\t\t\tAverage PSNR Anatomy: {psnr_anatomy}")
        print(f"\t\t\tAverage PSNR Characteristics: {psnr_characteristics}")

        # Log the metrics' averages
        wandb.log(
            {
                "Average MSE Input": mse_input,
                "Average MSE Anatomy": mse_anatomy,
                "Average MSE Characteristics": mse_characteristics,
                "Average SSIM Input": ssim_input,
                "Average SSIM Anatomy": ssim_anatomy,
                "Average SSIM Characteristics": ssim_characteristics,
                "Average PSNR Input": psnr_input,
                "Average PSNR Anatomy": psnr_anatomy,
                "Average PSNR Characteristics": psnr_characteristics,
            })

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
    Run the inference.

    :param configuration: Configuration of the inference run.
    """

    # Initialize the weights & biases project with the given inference configuration
    wandb.init(project="unoranic+_inference", name=f"{configuration['dataset']}-{configuration['architecture']}", config=configuration)

    # Initialize the inference
    print("Initializing the inference ...")
    inference = Inference(configuration)

    # Run the inference
    print("Run the inference...")
    inference.run_inference()


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
        config['inference']['architecture'] = architecture

        main(config['inference'])

    # Finished code
    end_time = time.time()

    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))