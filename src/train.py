from src.helpers import random_training_example, category_from_output
from src.models.rnn import RNN
import time
import math
import torch


def get_prediction(input_tensor, hidden, model):
    """
    Returns a prediction and hidden state given an input tensor and initial hidden state
    :param input_tensor: tensor
    :param hidden: tensor
    :param model: model
    :return: output, hidden
    """
    output = None
    for i in range(input_tensor.size()[0]):
        output, hidden = model(input_tensor[i], hidden)
    return output, hidden


def update_model(input_tensor, category, model, criterion, learning_rate):
    """
    Updates the model by computing the loss between the output produced by the input tensor and the true category
    :param input_tensor: tensor
    :param category: tensor
    :param model: model
    :param criterion: torch criterion
    :param learning_rate: float
    :return: output, loss_value
    """
    model.zero_grad()
    # output, hidden = get_prediction(input_tensor, model.init_hidden(), model)
    hidden = model.init_hidden()
    model.zero_grad()

    output = None

    for i in range(input_tensor.size()[0]):
        output, hidden = model(input_tensor[i], hidden)

    loss = criterion(output, category)
    loss.backward()
    loss_value = loss.item()

    for p in model.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss_value


def train(data_dict, model, model_name, criterion, learning_rate, iterations, loss_checkpoint_frequency):
    """
    Trains a given model and returns the trained model and the training losses
    :param data_dict: dictionary
    :param model: torch.nn.Module
    :param model_name: string
    :param criterion: torch criterion
    :param learning_rate: float
    :param iterations: integer
    :param loss_checkpoint_frequency: integer
    :return: model, losses
    """
    print(f'Training {model_name} for {iterations} iterations...')

    keys = list(data_dict.keys())
    dict_size = len(keys)
    total_data = 0
    for key in keys:
        print(f'{key}: {len(data_dict[key])}')
        total_data += len(data_dict[key])
    print(f'Data dict: {dict_size} Entries: {total_data}')

    losses = []
    loss_average = 0
    val_average = 0
    for i in range(1, iterations+1):
        input_tensor, category_tensor = random_training_example(data_dict, is_val=False)
        output, loss_value = update_model(input_tensor, category_tensor, model, criterion, learning_rate)
        loss_average += loss_value

        if i % loss_checkpoint_frequency == 0:
            losses.append(loss_average/loss_checkpoint_frequency)
            print(f"Progress = {round(i/iterations*100, 2)}%, Loss = {round(loss_average, 2)}, Iteration {i}/{iterations}")
            loss_average = 0

        # Now do validation loss with is_val=True
        # (set learning rate to 0 to ensure we're only validating, not training)

        input_tensor, category_tensor = random_training_example(data_dict, is_val=True)
        output, val_loss = update_model(input_tensor, category_tensor, model, criterion, 0)
        val_average += val_loss

        if i % loss_checkpoint_frequency == 0:
            #val_losses.append(val_average/loss_checkpoint_frequency)
            print(f"Validation = {round(i/iterations*100, 2)}%, Loss = {round(val_average, 2)}, Iteration {i}/{iterations}")
            val_average = 0

    return model, losses
