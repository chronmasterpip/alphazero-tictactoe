from mcts import MCTS
from tictactoe import TicTacToe
import numpy as np
from tqdm import tqdm
import random
from keras import Sequential
import pickle
import torch
from model import TicTacToeModel

class AlphaZero:
    def __init__(self, model_handler : TicTacToeModel, game : TicTacToe, args, filelabel='model') -> None:
        self.model_handler=model_handler
        self.game=game
        self.args=args
        self.mcts = MCTS(game, args, model_handler)
        self.filelabel=filelabel

        self.labels=[]

    def selfPlay(self):
        turn_history=[] #list storing the gamestate and action probabilites from mcts
        player =1
        state = self.game.get_initial_state()

        while 1:
            #print(f'starting state: {state.reshape(3,3)}')
            canonical_state=state*player #always give the gamestate to model as if it were about to play
            #print(f'canonical state (player {player}): {canonical_state.reshape(3,3)}')
            action_probs=self.mcts.search(canonical_state)
            turn_history.append([canonical_state, np.array(action_probs), player])
            action = np.random.choice(self.game.get_action_size(), p = action_probs)

            state = self.game.get_next_state(state, action, -player)
            #print(f'after action: {state.reshape(3,3)}')
            is_terminal, value = self.game.is_terminated(state)

            value= np.abs(value) #dont care which player won since we are using canonical perspective

            if is_terminal:
                model_history = []
                for state_hist, action_prob_hist, player_hist in turn_history:
                    #TODO: update value conversion to be game agnostic
                    value_hist = value if player_hist == player else -value #update values to take player perspective into account
                    model_history.append( #arrange the turn history in a way that can be input into the NN
                        [
                        self.game.get_encoded_state(state_hist),
                        action_prob_hist,
                        np.array(value_hist)#np.array(value_hist)
                        ]
                    )
                return model_history
            
            # TODO: update this to be agnostic to game player count
            player=-player 

    def train(self, history):
        self.model_handler.fit_model(history)

    def learn(self):

        for iteration in range(self.args["alpha_zero_it"]):
            print(f'Alpha-zero iteration: {iteration}')
            it_history = []

            if not self.model_handler.tf:
                self.model_handler.model.eval() 
            for selfplay_it in range(self.args['selfplay_it']):
                f'\tselfplay iteration: {selfplay_it}'
                result=self.selfPlay()
                it_history += result
                self.labels.append(result)
            
            self.train(it_history)

            self.model_handler.save_model(self.filelabel, iteration)

            with open(f'torch_checkpoints/{self.filelabel}_labels_{iteration}', 'wb') as f:
                pickle.dump(self.labels, f)

            self.labels=[]

