# -*- coding: utf-8 -*-
import torch
import matplotlib.pyplot as plt

# Fonction à minimiser : f(x) = (x - 3)^2
def loss_function(x):
    return (x - 3) ** 2

# Fonction pour entraîner un modèle avec un optimiseur donné
def train_optimizer(optimizer_name, learning_rate=0.1, epochs=100):
    x = torch.tensor([5.0], requires_grad=True)  # Initialisation
    optimizers = {
        "GD": torch.optim.SGD([x], lr=learning_rate),
        "SGD": torch.optim.SGD([x], lr=learning_rate, momentum=0),  # Version classique sans momentum
        "Adam": torch.optim.Adam([x], lr=learning_rate)
    }
    
    optimizer = optimizers[optimizer_name]
    losses = []

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses

# Paramètres
epochs = 50
learning_rate = 0.1

# Entraînement avec chaque optimiseur
losses_gd = train_optimizer("GD", learning_rate, epochs)
losses_sgd = train_optimizer("SGD", learning_rate, epochs)
losses_adam = train_optimizer("Adam", learning_rate, epochs)

# Affichage des courbes de convergence
plt.figure(figsize=(8, 5))
plt.plot(losses_gd, label="GD (batch gradient descent)", linestyle="--")
plt.plot(losses_sgd, label="SGD (stochastic gradient descent)", linestyle="-.")
plt.plot(losses_adam, label="Adam", linestyle="-")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Comparaison des algorithmes d optimisation")
plt.legend()
plt.grid()
plt.show()
