import numpy as np
from model import TicTacToeModel
from tensorflow import convert_to_tensor
import torch

class Node:
    def __init__(self, game, player=1, state=None, p=0.5, action_taken=None) -> None:
        self.children= []
        self.parent = None
        self.player=player
        self.action_taken=action_taken
        self.game = game

        if isinstance(state, np.ndarray):
            self.state=state
        else:    
            self.state=np.zeros(self.game.get_action_size())


        self.n=float(0) # number of times action has been taken
        self.w=float(0) # total value of the state (e.g. number of times this state was part of a win)
        self.q=float(0) # mean value of the state: q=w/n (wins per time node was visited)
        self.p=p # policy function prediction
        self.v=float(0) # value function prediction
        self.N=float(0) # total number of trials

    def is_expandable(self):
        return len(self.children)==0
        
    def __repr__(self):
        board= self.state.reshape(3,3)
        return np.array_str(board)
        #return np.array_str(board).replace("-1", 'O').replace("1", "X").replace("0", "-")
    
class MCTS:
    def __init__(self,game, args, model_handler : TicTacToeModel, verbose=False) -> None:
        self.c=args["c"]
        self.game=game
        self.args=args
        self.model_handler=model_handler
        self.verbose=verbose

    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, state=state.copy())
        
        for i in range(self.args['MCTS_it']):
            node = root
            while node.is_expandable() == False:
                if self.verbose:
                    for child in node.children:
                        print(f'child{child.action_taken} UCB: {self.UCB(child)}, visits {child.n}, value {child.w}')
                node=self.get_next_node(node)
                if self.verbose:
                    print(f'TRAVERSING node:\n {node}')
                    print(f'\tvisits: {node.n}, value: {node.w}')
            
            model_input=self.game.get_encoded_state(node.state)
            input_formatted=self.model_handler.get_formatted_input(model_input)
            predictions, model_value = self.model_handler.get_outputs(input_formatted) #model output has an empty nested list for some reason
            terminal, value=self.game.is_terminated(node.state)
            value= -np.abs(value)
            
            if not terminal:
                value=model_value
                predictions *= self.game.get_valid_actions(node, get_array=True)
                predictions /= np.sum(predictions)
                self.game.expand(node, predictions)
                if self.verbose:
                    print(f'LEAF node:\n {node}')
                    print(f'\tis terminal? {terminal}')
                    #print(f'\tbackprop value: {model_value}')
                    print(f'\tpredictions: {predictions}')
            if self.verbose:
                print(f'backpropping: {model_value}')
            self.backprop(node, value)
            
        action_probs=np.zeros(self.game.get_action_size())
        for child in root.children:
            action_probs[child.action_taken] = child.n
            #print(f'child{child.action_taken} visits: {child.n}')
        if self.verbose:
            print(f'un_normalized action probs: {action_probs}')
        action_probs *= self.game.get_valid_actions(root, get_array=True)
        action_probs /= np.sum(action_probs)

        if self.verbose:
            print(f'root value: {root.w}')
            print(f'root visits: {root.n}')
            print(f'action probs: {action_probs}')

        return action_probs


    def backprop(self, node, value) -> None:
        node.w += value #no longer multiplying by node.player
        node.n += 1

        if node.parent is not None:
            self.backprop(node.parent, -value)

    def UCB(self, node : Node) -> float:
        if node.n >0:
            q_value=(1-float(node.w)/float(node.n))*1/2
        else: 
            q_value=0
        
        return q_value + self.c*node.p*np.sqrt(node.parent.n)/(node.n+1)
        
    def get_next_node(self, node :Node) -> Node: 
        max_ucb=-np.inf
        chosen_child=node.children[0]
        for child in node.children:
            new_ucb= self.UCB(child)
            if new_ucb > max_ucb:
                chosen_child=child
                max_ucb=new_ucb

        return chosen_child