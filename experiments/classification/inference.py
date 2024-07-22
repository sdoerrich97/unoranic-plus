"""
xAILAB Bamberg
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Inference of unORANIC+ for a classification objective.

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
from torchmetrics import StructuralSimilarityIndexMeasure
from tqdm import tqdm

# Import own scripts
from data_loader import DataLoaderCustom
from ...models.classifier import Classifier
from ...utils import Metrics


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

        # Create the path to where the trained classifier shall be stored
        if self.architecture_encoder == 'resnet18' or self.architecture_encoder == 'resnet50' or self.architecture_encoder == 'ViT':
            self.checkpoint_file = self.input_path / f"{self.latent_dim}" / self.dataset / self.architecture_encoder / "checkpoint_best.pt"
            self.output_path = self.output_path / f"{self.latent_dim}" / self.dataset / self.architecture_encoder

        else:
            self.checkpoint_file = self.input_path / f"{self.latent_dim}" / self.dataset / self.architecture_encoder / self.architecture_decoder / "checkpoint_best.pt"
            self.output_path = self.output_path / f"{self.latent_dim}" / self.dataset / self.architecture_encoder / self.architecture_decoder

        self.output_path.mkdir(parents=True, exist_ok=True)

        # Enable the use of cudnn
        torch.backends.cudnn.benchmark = True

        # Initialize the dataloader
        print("\tInitializing the dataloader...")
        self.data_loader = DataLoaderCustom(self.dataset, self.data_path, self.img_size, self.batch_size, train=False)
        self.train_loader = self.data_loader.get_train_loader()
        self.val_loader = self.data_loader.get_val_loader()
        self.test_loader = self.data_loader.get_test_loader()

        # Initialize the model
        print("\tInitializing the model ...")
        self.checkpoint = torch.load(self.checkpoint_file, map_location='cpu')

        if self.architecture_encoder == 'resnet18' or self.architecture_encoder == 'resnet50':
            self.model = Classifier(self.img_size, self.in_channel, self.latent_dim, self.architecture_encoder,
                                    self.architecture_decoder, self.task, self.nr_classes)

        elif self.architecture_encoder == 'unoranic':
            self.latent_dim = 256  # Change the hyperparamters for the unoranic model
            self.model = Classifier(self.img_size, self.in_channel, self.latent_dim, self.architecture_encoder,
                                    self.architecture_decoder, self.task, self.nr_classes, dropout=self.dropout)

        elif self.architecture_encoder == 'ViT' or self.architecture_encoder == 'unoranic+':
            self.model = Classifier(self.img_size, self.in_channel, self.latent_dim, self.architecture_encoder,
                                    self.architecture_decoder, self.task, self.nr_classes, patch_size=self.patch_size,
                                    depth=self.depth, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
                                    norm_layer=self.norm_layer)

        else:
            raise SystemExit

        self.model.load_state_dict(self.checkpoint['model'])
        self.model = self.model.to(self.device)
        self.model.requires_grad_(False)

        # Initialize the loss criterion
        print("\tInitialize the loss criterion...")
        if self.task == "multi-label, binary-class":
            self.loss_criterion = nn.BCEWithLogitsLoss().to(self.device)
        else:
            self.loss_criterion = nn.CrossEntropyLoss().to(self.device)

    def run_inference(self):
        """
        Run inference on the content encoder-decoder pair.
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
                self.run_inference_for_single_set(set_loader, nr_batches)

            # Stop the time
            end_time_set = time.time()
            hours_set, minutes_set, seconds_set = self.calculate_passed_time(start_time_set, end_time_set)

            print("\t\t\tElapsed time for set '{}': {:0>2}:{:0>2}:{:05.2f}".format(set, hours_set, minutes_set, seconds_set))

        # Stop the time
        end_time = time.time()
        hours, minutes, seconds = self.calculate_passed_time(start_time, end_time)

        print("\t\t\tElapsed time for inference: {:0>2}:{:0>2}:{:05.2f}".format(hours, minutes, seconds))

    def run_inference_for_single_set(self, set_loader, nr_batches):
        """
        Run the inference for the classifier model.

        :param set_loader: Data loader for the current set.
        :param nr_batches: Number of batches to process.
        :param output_path: Where the image shall be stored.
        """

        # Initialize a progress bar
        pbar = tqdm(total=len(set_loader), bar_format='{l_bar}{bar}', ncols=80, initial=0, position=0, leave=False)
        pbar.set_description(f'\t\t\tProgres')

        loss = 0
        Y_target, Y_predicted = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)

        for i, (_, Y, X, _) in enumerate(set_loader):
            # Map the input to the respective device
            if self.task == "multi-label, binary-class":
                X, Y = X.to(self.device), Y.to(torch.float32).to(self.device)
            else:
                X, Y = X.to(self.device), Y.to(self.device).reshape(-1)

            # Run the input through the model
            Z = self.model(X)

            # Calculate the loss
            loss_batch = self.loss_criterion(Z, Y)

            print(f"\t\t\tBatch: [{i}/{nr_batches}]")
            print(f"\t\t\t\tCrossEntropyLoss: {loss_batch}")

            # Store the target and predicted labels
            loss += loss_batch.detach().cpu()
            Y_target = torch.cat((Y_target, Y.detach()), dim=0)
            Y_predicted = torch.cat((Y_predicted, Z.detach()), dim=0)

            # Update the progress bar
            pbar.update(1)

        # Average the loss value across all batches and compute the performance metrics
        loss /= nr_batches

        ACC = Metrics.getACC(Y_target.cpu().numpy(), Y_predicted.cpu().numpy(), self.task)
        AUC = Metrics.getAUC(Y_target.cpu().numpy(), Y_predicted.cpu().numpy(), self.task)

        # Print the loss values and send them to wandb
        print(f"\t\t\tAverage Cross-Entropy (CE) Loss: {loss}")
        print(f"\t\t\tAverage Accuracy (ACC): {ACC}")
        print(f"\t\t\tAverage Area-Under-The-Curve (AUC): {AUC}")

        # Log the metrics' averages
        wandb.log(
            {
                "Average Cross-Entropy (CE) Loss": loss,
                "Average Accuracy (ACC)": ACC,
                "Average Area-Under-The-Curve (AUC)": AUC
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
    wandb.init(project="unoranic+_classification_inference", name=f"{configuration['dataset']}-{configuration['architecture_encoder']}", config=configuration)

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
    parser.add_argument("--architecture_encoder", required=False, default='unoranic+', type=str, help="Which model to use for the encoder.")
    parser.add_argument("--architecture_decoder", required=False, default='single_layer', type=str, help="Which model to use for the decoder.")

    args = parser.parse_args()
    config_file = args.config_file
    architecture_encoder = args.architecture_encoder
    architecture_decoder = args.architecture_decoder

    # Load the default and the hyperparameters and run the training
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        config['inference']['architecture_encoder'] = architecture_encoder
        config['inference']['architecture_decoder'] = architecture_decoder

        main(config['inference'])

    # Finished code
    end_time = time.time()

    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))