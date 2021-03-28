import string
import torch


class Directories:
    """
    Contains the paths of all of the directories referenced in the project
    """
    data = "data/"
    results = "results/"
    models = "models/"


class Globals:
    """
    Maintains global constants
    """
    ALL_LETTERS = string.ascii_letters + " .,;'"
    N_LETTERS = len(ALL_LETTERS)
    N_CATEGORIES = 18


class RNNParams:
    """
    Contains all of the constants associated with the RNN model
    """
    N_HIDDEN = 128

class TrainingParams:
    """
    Contains the parameters used to train the model
    """
    learning_rate = 0.005
    criterion = torch.nn.NLLLoss()
    loss_checkpoint_frequency = 5000
    iterations = 100000
