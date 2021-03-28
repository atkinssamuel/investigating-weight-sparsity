from src.constants import Directories, Globals
from src.helpers import read_lines
import glob
import os


def import_names_data():
    """
    Imports the data stored in the data folder and returns a data_dict dictionary

    data_dict: dictionary where the keys are the nationalities ("Scottish", "French", ...) and the values are the names
    that belong to that given nationality ("Scottish": ["Smith", "Brown", "Wilson", ...])

    :return: data_dict
    """

    names_data_files = Directories.data + "names_data/names/*.txt"
    data_dict = {}

    for file in glob.glob(names_data_files):
        name_category = os.path.splitext(os.path.basename(file))[0]
        lines = read_lines(file)
        data_dict[name_category] = lines

    return data_dict
