import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import norm

sns.set()


def norm_variance(arr: np.ndarray):
    assert len(arr.shape) == 3, arr.shape
    return np.mean(
        norm(arr - np.mean(arr, axis=1, keepdims=True), ord=2, axis=-1) ** 2,
        axis=1,
    )


def max_variance(arr: np.ndarray):
    assert len(arr.shape) == 3, arr.shape
    return np.max(
        norm(arr - np.mean(arr, axis=1, keepdims=True), ord=2, axis=-1) ** 2,
        axis=1,
    )


filename = "outputs/bounce_grads_40.npz"
print("Loading", filename)

data = np.load(filename)
fobgs = data["fobgs"]
zobgs = data["zobgs"]
loss = data["losses"]
baseline = data["baseline"]
zobgs = np.nan_to_num(zobgs)
zobgs_no_grad = data["zobgs_no_grad"] if "zobgs_no_grad" in data else None
if zobgs_no_grad is not None:
    print("Found no grad!")

if hasattr(data, "h"):
    hh = data["h"]
    H = len(hh)
else:
    H = zobgs.shape[0]
    hh = np.arange(H)

N = zobgs.shape[1]
th_dim = zobgs.shape[2]
n = data["n"]
m = data["m"]
std = data["std"]
print(f"Loaded grads with H={H}, N={N} n={n} m={m} std={std}")

grad_names = []
for j in range(m):
    grad_names.extend([f"zobgs_{j}", f"fobgs_{j}", f"zobgs_no_grad_{j}"])
columns = ["H", "loss"] + grad_names
df = pd.DataFrame(columns=columns)

for i in range(H):
    d = {"H": i + 1, "loss": loss[i]}
    for j in range(m):
        d.update({f"zobgs_{j}": zobgs[i, :, j], f"fobgs_{j}": fobgs[i, :, j]})
        if zobgs_no_grad is not None:
            d.update({f"zobgs_no_grad_{j}": zobgs_no_grad[i, :, j]})
    df = pd.concat((df, pd.DataFrame(d)))
df = df.explode(["loss"] + grad_names)
df = df.reset_index()

print("Plotting")
f, ax = plt.subplots(2, 2, figsize=(12, 8))
f.suptitle(filename.replace(".npz", ""))

# 1. Plot bias
diff = zobgs.mean(axis=1) - fobgs.mean(axis=1)
bias_l2 = norm(diff, ord=2, axis=-1)
bias_l1 = norm(diff, ord=1, axis=-1)
ax[0, 0].plot(hh, bias_l2, label="L2 Bias")
ax[0, 0].plot(hh, bias_l1, label="L1 Bias")
ax[0, 0].set_title("FoBG bias wrt ZoBG")
ax[0, 0].legend()
ax[0, 0].set_xlabel("H")

# 2. Plot gradient estiamtes
for j in range(m):
    sns.lineplot(
        df, x="H", y=f"zobgs_{j}", ax=ax[1, 0], errorbar="sd", label=f"ZoBGs {j}"
    )
    sns.lineplot(
        df, x="H", y=f"fobgs_{j}", ax=ax[1, 0], errorbar="sd", label=f"FoBGs {j}"
    )
    if zobgs_no_grad is not None:
        sns.lineplot(
            df,
            x="H",
            y=f"zobgs_no_grad_{j}",
            ax=ax[1, 0],
            errorbar="sd",
            label=f"True ZoBGs {j}",
        )
ax[1, 0].set_title("Gradient estimate wrt action")
ax[1, 0].set_ylabel(None)
ax[1, 0].legend()

# 3. Plot variance
ax[0, 1].plot(hh, norm_variance(zobgs), label="ZoBGs")
ax[0, 1].plot(hh, norm_variance(fobgs), label="FoBGs")
if zobgs_no_grad is not None:
    ax[0, 1].plot(hh, norm_variance(zobgs_no_grad), label="True ZoBGs")
ax[0, 1].plot(hh, hh**3 * m / (N * std**2), label="Lemma 3.10")
ax[0, 1].set_yscale("log")
ax[0, 1].set_xlabel("H")
ax[0, 1].set_title("Gradient variance")
ax[0, 1].legend()

# 4. Plot loss
sns.lineplot(df, x="H", y="loss", ax=ax[1, 1], errorbar="sd", label="Loss")
ax[1, 1].plot(hh, baseline, label="Baseline")
ax[1, 1].legend()

plt.tight_layout()
filename = filename.replace(".npz", ".pdf")
print("Saving to {:}".format(filename))
plt.savefig(filename)
