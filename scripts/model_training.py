# This part of the training is to be carried out continuously.
# Each time you run the cells below, the last model will be loaded and training will continue from that model.
# It is therefore important not to run the first part of the training every time, otherwise the model will start from scratch.

# The code below is designed to run for as long as you like.
# It can be stopped at any time, as checkpoints are built in
# and you can choose the number of epochs performed between each checkpoint
# by modifying the `SAVE_PERIOD` variable (5 by default).

SAVE_PERIOD = 5

import os
import torch
from ultralytics import YOLO
from datetime import datetime

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(PROJECT_PATH, 'models')
CONFIG_PATH = os.path.join(PROJECT_PATH, 'data', 'augmented_train', 'data.yaml')

def load_last_model(models_dir):
    # models are saved using the date : `YYYY-MM-DD_HHhMM_yolo11n.pt`
    # we can sort them and keep the last one to get the last trained model
    model_name = sorted([f for f in os.listdir(models_dir) if f.endswith('.pt')])[-1]
    model = YOLO(os.path.join(models_dir, model_name))
    
    assert model is not None, 'Model not found'
    assert type(model) == YOLO, 'Model is not a YOLO model'
    print(f'\u2705 Model loaded: {model_name}')
    return model

device = torch.device(
    'cuda' if torch.cuda.is_available()              # cuda for NVIDIA GPUs
    else 'mps' if torch.backends.mps.is_available()  # mps for Apple silicon
    else 'cpu')                                      # cpu otherwise

assert device is not None, 'Device not found'
assert type(device) == torch.device, 'Device is not a torch device'
assert device.type in ['cuda', 'mps', 'cpu'], 'Device is not a valid torch device'
print(f'Using device: {device}')

current_epoch = 0
LAUNCH_DATE = datetime.now().strftime('%Y-%m-%d_%Hh%M')

try:
    while True:
        print(f'Epoch {current_epoch+1} to {current_epoch+SAVE_PERIOD}...')
        
        model = load_last_model(MODELS_DIR)
        results = model.train(
            data=CONFIG_PATH,
            epochs=SAVE_PERIOD,  # training on defined number of epochs
            device=device
        )
        
        current_epoch += SAVE_PERIOD
        model_path = os.path.join(MODELS_DIR, f'{LAUNCH_DATE}_epoch{current_epoch}_yolo11n.pt')
        model.save(model_path)
        print(f'Model saved: {model_path}')
        
except KeyboardInterrupt:
    print('Training interrupted by the user')
    try:
        print(f'Last model saved: {model_path}')
    except:
        print('No model was saved during this session')