import numpy as np
from neural_network import NeuralNetwork
from genetic import GeneticTrainer
import matplotlib.pyplot as plt

x_train = np.linspace(0, 2 * np.pi, 100)
y_train = np.sin(x_train)

train_data = [(np.array([x]), np.array([y])) for x, y in zip(x_train, y_train)]

def evaluate_network(net):
    preds = [net.propagate(x)[0] for x, _ in train_data]
    targets = [y[0] for _, y in train_data]
    mse = np.mean((np.array(preds) - np.array(targets)) ** 2)
    return -mse  

template_net = NeuralNetwork(
    layers_sizes=[1, 8, 1],
    activations=["sigmoid", "identity"],  
    loss_name="mse"
)

trainer = GeneticTrainer(
    network=template_net,
    population_size=200,
    mutation_rate=0.1,
    elite_fraction=0.2
)

generations = 1000
for gen in range(generations):
    scores = trainer.evaluate_population(evaluate_network)
    best_score = max(scores)
    print(f"Generacja {gen+1}/{generations} | Najlepszy -MSE: {-best_score:.6f}")
    trainer.evolve(scores)a

best_net = trainer.population[np.argmax(scores)]

x_test = np.linspace(0, 2 * np.pi, 100)
y_true = np.sin(x_test)
y_pred = [best_net.propagate([x])[0] for x in x_test]

plt.plot(x_test, y_true, label="sin(x)", linewidth=2)
plt.plot(x_test, y_pred, label="Neural Network", linestyle='--')
plt.title("Learning sin(x) using Genetic Algorithm")
plt.legend()
plt.grid(True)
plt.show()
