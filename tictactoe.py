import numpy as np
from mcts import Node
import copy

class TicTacToe:
    def __init__(self, args,  dim=3) -> None:
        self.dim=dim
        self.args=args

    def get_initial_state(self):
        return np.zeros(self.dim**2)
    
    def get_action_size(self):
        return self.dim**2
        
    def get_valid_actions(self, node : Node, get_array=False):
        if get_array:
            r_state=np.zeros(self.dim**2)
            r_state[np.where(node.state==0)[0]]=1
            return r_state
        else:
            return np.where(node.state==0)[0]
    
    def is_terminated(self, state):
        '''

        returns False if game is still going, True if player has won, -1 if game is a draw
        '''
        #game is concluded if one player has 3 consecutive marks

        board=copy.copy(state).reshape(3,3)
        #check for row wins
        r_sum=[np.sum(row) for row in np.split(board,self.dim)]
        #check for column wins
        c_sum=[np.sum(column) for column in np.split(board,self.dim, axis=1)]
        #check for diag wins
        diag_sum=[
            np.sum(np.diagonal(board)), #check the main diagonal
            np.sum(np.diagonal(np.fliplr(board))) #reverse board rows, then check the main diagonal
        ]

        totals=r_sum+c_sum+diag_sum
        if self.dim in totals:
            return True, +1
        elif -self.dim in totals:
            return True, -1
        elif len(np.where(state == 0)[0]) == 0:
            return True, 0
        else:
            return False, 0
        
    def get_model_input(self, state : Node):
        # return node.state.reshape(3,3)
        return np.expand_dims(self.get_encoded_state(state), axis=0)
    
    def get_encoded_state(self, state : Node):
        player_state=np.zeros(self.dim**2)
        player_state[np.where(state==1)[0]]=1

        opponent_state=np.zeros(self.dim**2)
        opponent_state[np.where(state==-1)[0]]=1

        empty_state=np.zeros(self.dim**2)
        empty_state[np.where(state==0)[0]]=1
    
        return np.array([opponent_state.reshape(3,3), empty_state.reshape(3,3), player_state.reshape(3,3)]).astype(np.float32)
    

    def expand(self, node : Node, predictions):
        for ind, action in enumerate(self.get_valid_actions(node)):
            child=self.create_child_node(node, action, predictions[action]) #was predictions[ind].. holy shit
            node.children.append(child)

    def create_child_node(self, node, action, prob):
        new_state = node.state.copy()
        new_state=new_state*-1 # change perspective
        new_state[action] = -1
        child=Node(game=self, state=new_state, p=prob, player=-node.player)
        child.parent=node
        child.action_taken=action
        return child

    def get_child_node(self, node, action):
        #get the child corresponding to action
        for child in node.children:
            difference=node.state-child.state

            if (np.where(difference==1 )[0] == action) ==1:
                return child
        
        return None
    
    def get_next_state(self, state, action, last_player):
        '''
        args:
            state: state to be acted on
            action: action to be taken as an int
            last_player: player who most recently played. 
            
        Note the value to be added to the board will be -player. 
        '''
        new_state = copy.copy(state)
        new_state[action]=-last_player
        return new_state
        