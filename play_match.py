from tictactoe import TicTacToe
from mcts import Node, MCTS
import numpy as np
from model import TicTacToeModel
from alpha_zero import AlphaZero

import torch

args = {
    'c': 2,
    'MCTS_it': 1000
}
tictactoe = TicTacToe(args)
player = 1



model_handler=TicTacToeModel(args, tf=False)
model=model_handler.model
model.load_state_dict(torch.load('torch_checkpoints/torch_model_noseed_2.pt'))
model.eval()

mcts = MCTS(tictactoe, args, model_handler)


play_again=True
while play_again:
    state = tictactoe.get_initial_state()
    while True:
        
        print(state.reshape(3,3))
        
        if player == 1:
            valid_moves = tictactoe.get_valid_actions(Node(tictactoe, state=state), get_array=True )
            print("valid_moves", [i for i in range(tictactoe.get_action_size()) if valid_moves[i] == 1])
            action = int(input(f"{player}:"))

            if valid_moves[action] == 0:
                print("action not valid")
                continue
                
        else:
            canonical_state = state * -1
            action_probs = mcts.search(canonical_state)
            action = np.argmax(action_probs)
            
        state = tictactoe.get_next_state(state, action, -player)
        
        is_terminal, value = tictactoe.is_terminated(state)
        value=np.abs(value)

        if is_terminal:
            print(state.reshape(3,3))
            if value == 1 and player==1:
                print(f"player wins!")
            elif value ==1 and player ==-1:
                print(f"model wins!")
            else:
                print(f'draw')

            player_input=input('play again? (y/n)')
            if player_input.startswith('n'):
                play_again=False
            break
            
        player = -player