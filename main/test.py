#%%
import numpy as np
import matplotlib.pyplot as plt
a = np.linspace(-10, 10, 1000)
def sigmoid(x, k):
    return 1 / (1 + np.exp(-k * x))
plt.figure(figsize=(7, 5))
for k in [2, 1, 0.5, 4]:
    plt.plot(a, sigmoid(a, k), label=f"k = {k}")
plt.xlabel("a")
plt.ylabel("sigmoid(a)")
plt.title("Sigmoid functions with different slopes")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()