# requirements
import os
import ast
import pandas
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

base_dir = Path(__file__).resolve().parent.parent.parent
load_dotenv(f'{base_dir}/.env')
data_path = os.getenv('BASE_DATA')

# train_path=os.getenv(f'TRAIN_DATA_DIR')
# test_path= 
# metadata_path=

# print(train_data.columns)



def load_channel(chan_id,data_dir=data_path):
