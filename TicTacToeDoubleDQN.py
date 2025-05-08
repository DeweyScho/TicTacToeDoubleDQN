#Dewey Schoenfelder
import os
import time
import torch
import torch.nn.functional as F
import sys
from train import train
from TicTacToe import TicTacToe
from Network import Network
import numpy as np


def play(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TicTacToe()
    done = False
    obs = env.reset()
    exp = {}
    player = 0

    while not done:
        time.sleep(1)
        print("Commands:\n{}|{}|{}\n-----\n{}|{}|{}\n-----\n{}|{}|{}\n\nBoard:".format(*[x for x in range(0, 9)]))
        env.render()  # display current state
        action = None

        # human player (player 1) is displayed as state 2 in the cell
        # computer (player 0) is displayed as 1 on screen, empty cell as 0
        if player == 1:
            action = int(input())  # human player's input for cell number
        else:
            time.sleep(1)  # trained network's decision
            action = act(model, torch.tensor(np.array([obs]), dtype=torch.float).to(device)).item()

        obs, _, done, exp = env.step(action)
        player = 1 - player  # change player turn from 0 to 1 or 1 to 0

        os.system("cls" if os.name == "nt" else "clear")

        print("Commands:\n{}|{}|{}\n-----\n{}|{}|{}\n-----\n{}|{}|{}\n\nBoard:".format(*[x for x in range(0, 9)]))
        env.render()
        print(exp)

        if "reason" in exp and "tied" in exp["reason"]:
            print("A tied game. ---------.")
            exit(0)


def load_model(path: str, device: torch.device):
    model = Network(n_inputs=3 * 9, n_outputs=9).to(device)
    checkpoint_data = torch.load(path)
    model.load_state_dict(checkpoint_data['state_dict'])
    model.eval()
    return model


def act(model: Network, state: torch.Tensor):
    with torch.no_grad():
        p = F.softmax(model.forward(state), dim=-1).cpu().numpy()[0]  # 9 outputs

        # find non-empty cells
        valid_moves = (state.cpu().numpy().reshape(3, 3, 3).argmax(axis=2).reshape(-1) == 0)

        p = valid_moves * p  # 9 numbers with 0 where the cell is not empty
        return torch.tensor(p).argmax()


def main():
    # Uncomment the line below to train the model
    res = train()

    # after model is trained, use the following code to play
    # human is player 2, select the cell number 0-8 to decide your turn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model('checkpoint/tictactoe_policy_model.pt', device)
    play(model)


if __name__ == "__main__":
    sys.exit(int(main() or 0))
