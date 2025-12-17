import json
import matplotlib.pyplot as plt

with open("models/training_metrics.json") as f:
    data = json.load(f)

epochs = range(1, len(data["train_loss"]) + 1)

plt.figure()
plt.plot(epochs, data["train_loss"], label="Train Loss")
plt.plot(epochs, data["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, data["train_acc"], label="Train Accuracy")
plt.plot(epochs, data["val_acc"], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch")
plt.legend()
plt.show()
