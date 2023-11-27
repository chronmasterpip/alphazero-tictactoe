


from tictactoe import TicTacToe
from mcts import Node, MCTS
import numpy as np
from model import TicTacToeModel
from alpha_zero import AlphaZero
import pickle
import datetime
from keras.models import load_model
import torch 

#model=load_model('model_checkpoints/refactor_1.keras')
#print(f'previous model loaded, beginning training')


args={"c" : 2, "MCTS_it" : 60, "selfplay_it" : 500, "alpha_zero_it": 3, "epochs" : 4, "batch_size" : 64}

model_handler=TicTacToeModel(args, tf=False)
#model_handler.model=load_model('model_checkpoints/refactor_1.keras')

#model = model_handler.model
#model.load_state_dict(torch.load('torch_checkpoints/torch_model_canonical_persp_1.pt'))

#optimizer= model_handler.optimizer
#optimizer.load_state_dict(torch.load('torch_checkpoints/optimizer_canonical_persp_0.pt'))

filelabel='noseed'
az=AlphaZero(model_handler, TicTacToe(args), args, filelabel=filelabel)

start=datetime.datetime.now()
az.learn()
end=datetime.datetime.now()
print(f'total time: {end-start}')