# geneticNN

**geneticNN** is a simple neural network framework trained using a genetic algorithm. The implementation is written in pure Python and uses only `numpy` — no external machine learning libraries are used.

## Overview

This project includes:
- A custom feedforward neural network (`neural_network.py`)
- Genetic algorithm for evolving network weights (`genetic.py`)
- Example: approximating the sine function (`learning_functions.py`)

The neural network is fully customizable. You can define:
- Layer sizes
- Activation functions per layer (ReLU, sigmoid, tanh, etc.)
- Fitness function and training logic

Everything is implemented from scratch — including forward propagation and genetic optimization (selection, crossover, mutation).

## Requirements

- Python 3.x
