from src.import_data import import_names_data
from src.train import train
from src.models.rnn import RNN
from src.constants import TrainingParams
from src.helpers import evaluate_model, save_model


if __name__ == "__main__":
    """
    data_dict: dictionary where the keys are the nationalities ("Scottish", "French", ...) and the values are the names
    that belong to that given nationality ("Scottish": ["Smith", "Brown", "Wilson", ...])
    """
    data_dict = import_names_data()

    # specifying the model and the model name
    rnn = RNN()
    model_name = "RNN"

    train_flag = True
    if train_flag:
        model, losses = train(data_dict=data_dict, model=rnn, model_name=model_name, criterion=TrainingParams.criterion,
                              learning_rate=TrainingParams.learning_rate, iterations=TrainingParams.iterations,
                              loss_checkpoint_frequency=TrainingParams.loss_checkpoint_frequency)
        evaluate_model(model, model_name, losses)
        save_model(model, model_name)
