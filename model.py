from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
import random
import tqdm

class TicTacToeModel:

    def __init__(self, args, num_chanels=512, dropout=0.3, dim=3, lr=0.01, tf=True) -> None:
        self.model=None
        self.args=args
        self.num_chanels=num_chanels
        self.dropout=dropout
        self.dim=dim
        self.lr=lr
        self.tf=tf
        self.optimizer=None

        if self.tf:
            self.init_tf_model()
        else:
            self.init_torch_model()


    def init_tf_model(self):

        action_size = self.dim**2

        num_channels=self.num_chanels
        dropout=self.dropout

        # Neural Net
        input_boards = Input(shape=(self.dim, self.dim, 3))    # s: batch_size x board_x x board_y

        x_image = (input_boards)                # batch_size  x board_x x board_y x 1
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(num_channels, 3, padding='same')(x_image)))         # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(num_channels, 3, padding='same')(h_conv1)))         # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(num_channels, 3, padding='same')(h_conv2)))        # batch_size  x (board_x) x (board_y) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(num_channels, 3, padding='valid')(h_conv3)))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4_flat = Flatten()(h_conv4)       
        s_fc1 = Dropout(dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc1))))          # batch_size x 1024
        pi = Dense(action_size, activation='softmax', name='pi')(s_fc2)   # batch_size x action_size
        v = Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1

        self.model = Model(inputs=input_boards, outputs=[pi, v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(self.lr))

    def init_torch_model(self):
        self.model=ResNet( 4, 64, dim=self.dim)
        self.optimizer=torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_formatted_input(self, input):
        if self.tf:
            return np.expand_dims(input, axis=0)
        else:
            return torch.tensor(input.astype(np.float32)).unsqueeze(0)

    def get_outputs(self, input):
        if self.tf:
            out=self.model(input)
            return [output.numpy()[0] for output in out]
        else:
            policy, value = self.model(
            input)
            policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()
            
            value = value.item()
            return [policy, value]
    
    def fit_model(self, history):
        if self.tf:
            #keras can automatically handle batching and epoch training
            random.shuffle(history)
            gamestates, policy_targets, value_targets = zip(*history)
            policy_targets=np.array(policy_targets).astype(np.float32)
            value_targets=np.array(value_targets).astype(np.float32)
            gamestates=np.array(gamestates).astype(np.float32).squeeze()
            self.model.fit(x=gamestates,  y=[policy_targets, value_targets], epochs=self.args['epochs'])
        else:
            #for pytorch need to manually batch and set up training epochs
            self.model.train()
            for epoch in tqdm.tqdm(range(self.args['epochs'])):
                random.shuffle(history)
                for batchIdx in range(0, len(history), self.args['batch_size']):
                    sample = history[batchIdx:min(len(history) - 1, batchIdx + self.args['batch_size'])] # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
                    
                    state, policy_targets, value_targets = zip(*sample)
                    
                    state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets)
                    
                    state = torch.tensor(state, dtype=torch.float32).squeeze()
                    policy_targets = torch.tensor(policy_targets, dtype=torch.float32)
                    value_targets = torch.tensor(value_targets, dtype=torch.float32).reshape(-1,1)

                
                    out_policy, out_value = self.model(state)
                    
                    policy_loss = F.cross_entropy(out_policy, policy_targets)
                    value_loss = F.mse_loss(out_value, value_targets)
                    loss = policy_loss + value_loss
                    print(f'Epoch {epoch}: policy loss: {policy_loss}, value loss: {value_loss}, loss: {loss}')
                    
                    self.optimizer.zero_grad() # change to self.optimizer
                    loss.backward()
                    self.optimizer.step() # change to self.optimizer
    
    def save_model(self, filelabel, iteration):
        if self.tf:
            self.model.save(f'model_checkpoints/{filelabel}_{iteration}.keras')
        else:
            torch.save(self.model.state_dict(), f"torch_checkpoints/torch_model_{filelabel}_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"torch_checkpoints/optimizer_{filelabel}_{iteration}.pt")


class ResNet(nn.Module):
    def __init__(self, num_resBlocks, num_hidden, dim):
        self.dim=dim
        super().__init__()
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * self.dim**2, self.dim**2)
        )
        
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * self.dim**2, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value
        
        
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
        