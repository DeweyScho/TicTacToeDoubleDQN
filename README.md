# TicTacToeDoubleDQN

A PyTorch implementation of the Double Deep Q-Network (Double DQN) algorithm applied to the game of Tic-Tac-Toe. The agent learns to play through self-play and updates Q-values using two neural networks to reduce overestimation bias.

Files:
- TicTacToeDoubleDQN.py — Main training loop
- TicTacToeEnv.py — Custom Gym-style Tic-Tac-Toe environment
- DQNetwork.py — Neural network for Q-value estimation
- ReplayMemory.py — Experience replay implementation
- checkpoint/ — Saved models

create checkpoint folder before training

Install dependencies:
pip install torch numpy matplotlib
