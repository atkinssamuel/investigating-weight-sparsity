from src.constants import Globals, Directories
import unicodedata
import os
import random
import torch
import matplotlib.pyplot as plt


def to_ascii(string):
    """
    Converts the input to ASCII encoding
    :param string: string input
    :return: string converted to ASCII
    """
    return "".join(char for char in unicodedata.normalize("NFD", string)
                   if unicodedata.category(char) != "Mn"
                   and char in Globals.ALL_LETTERS)


def read_lines(file):
    """
    Opens the provided file and reads each of the lines into an array
    :param file: file
    :return: array of lines
    """
    data = open(file, encoding='utf-8').read().strip().split('\n')
    return [to_ascii(line) for line in data]


def letter_to_index(letter):
    """
    Given a letter, returns an index associated with that letter based on the number of characters in the
    character-level environment
    :param letter: a letter ("A", "E", "d", etc.)
    :return: integer
    """
    return Globals.ALL_LETTERS.find(letter)


def letter_to_tensor(letter):
    """
    Encodes the input into a tensor using the letter_to_index function
    :param letter: a letter ("A", "E", "d", etc.)
    :return: integer tensor: an integer tensor with one of the values set to 1 and the rest 0
    """
    letter_tensor = torch.zeros(1, Globals.N_LETTERS)
    letter_tensor[0][letter_to_index(letter)] = 1
    return letter_tensor


def line_to_tensor(line):
    """
    Transforms a line into a 3-d tensor by iterating with the letter_to_tensor function
    :param line: a string ("William")
    :return: 3-d integer tensor
    """
    line_tensor = torch.zeros(len(line), 1, Globals.N_LETTERS)
    for i, letter in enumerate(line):
        line_tensor[i][0][letter_to_index(letter)] = 1
    return line_tensor


def category_from_output(data_dict, output):
    """
    Returns the category predicted by an output tensor
    :param data_dict: dictionary
    :param output: tensor
    :return: integer
    """
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return list(data_dict.keys())[category_i], category_i


def random_choice(array):
    """
    Returns a random element from the input array
    :param array: list
    :return: array[random index]
    """
    return array[random.randint(0, len(array) - 1)]


def random_training_example(data_dict):
    """
    Returns a random training example from the input data
    :param data_dict: dictionary
    :return: input, category
    """
    random_category = random_choice(list(data_dict.keys()))
    line = random_choice(data_dict[random_category])

    # category that the randomly sampled input tensor belongs to tensor
    category_tensor = torch.tensor([list(data_dict.keys()).index(random_category)], dtype=torch.long)
    # randomly sampled input tensor
    input_tensor = line_to_tensor(line)

    return input_tensor, category_tensor


def evaluate_model(model, model_name, losses):
    """
    Plots the losses and evaluates the model using a variety of metrics
    :param model: model
    :param model_name: string
    :param losses: list of floats
    :return: None
    """
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(Directories.results + f"{model_name}-losses.png")
    plt.clf()
    return


def save_model(model, model_name):
    """
    Saves the trained model
    :param model_name: string
    :param model: model
    :return: None
    """
    torch.save(model.state_dict(), Directories.models + model_name)
    return


def load_model(model, model_name):
    """
    Given an input model and a model path, loads a PyTorch model
    :param model: model
    :param model_name: name of the model
    :return: model
    """
    return model.load_state_dict(torch.load(Directories.models + model_name))