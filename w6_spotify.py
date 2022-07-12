import os
from configparser import ConfigParser
### SAVE MODEL AND SCALER
import pickle

def load(filename = "filename.pickle"):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)

    except FileNotFoundError:
        print("File not found!")




load()