import matplotlib.pyplot as plt
from constants import EPOCHS


def plot_loss_curves(results: dict[str, list[float]]) -> None:
    train_loss = results["train_loss"]
    test_loss = results["test_loss"]
    test_acc = results['test_acc']
    train_acc = results['train_acc']
    epochs = range(EPOCHS)
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, test_acc, label="test_acc")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()