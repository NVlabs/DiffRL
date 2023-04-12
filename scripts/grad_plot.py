import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

filename = "CartPoleSwingUpEnv_grads_200.npz"
print("Loading", filename)

data = np.load(filename)
fobgs = data["fobgs"]
zobgs = data["zobgs"]
loss = data["losses"]
baseline = data["baseline"]
zobgs = np.nan_to_num(zobgs)
zobgs_analytical = data["zobgs_analytical"] if "zobgs_analytical" in data else None
zobgs_no_grad = data["zobgs_no_grad"] if "zobgs_no_grad" in data else None
if zobgs_no_grad is not None:
    print("Found no grad!")

H = zobgs.shape[0]
N = zobgs.shape[1]
th_dim = zobgs.shape[2]
n = data["n"]
m = data["m"]
std = data["std"]
print(f"Loaded grads with H={H}, N={N} n={n} m={m} std={std}")

columns = ["H", "loss", "zobgs", "fobgs", "zobgs_no_grad"]
df = pd.DataFrame(columns=columns)

# TODO parametarize
th_idx = 0
print("Plotting for gradients with index", th_idx)
# TODO I think this is actully wrong. Should handle this as vectors not scalars
zobgs = zobgs[:, :, th_idx]
fobgs = fobgs[:, :, th_idx]
if zobgs_no_grad is not None:
    zobgs_no_grad = zobgs_no_grad[:, :, th_idx]

for i in range(H):
    d = {"H": i + 1, "loss": loss[i], "zobgs": zobgs[i], "fobgs": fobgs[i]}
    if zobgs_no_grad is not None:
        d.update({"zobgs_no_grad": zobgs_no_grad[i]})
    df = pd.concat((df, pd.DataFrame(d)))
df = df.explode(["loss", "zobgs", "fobgs", "zobgs_no_grad"])
df = df.reset_index()

print("Plotting")
f, ax = plt.subplots(2, 2, figsize=(12, 8))
f.suptitle(filename.replace(".npz", ""))

# 1. Plot bias
hh = np.arange(H) + 1
bias_l2 = ((zobgs.mean(axis=1) - fobgs.mean(axis=1)) ** 2) ** 0.5
bias_l1 = np.abs(zobgs.mean(axis=1) - fobgs.mean(axis=1))
ax[0, 0].plot(hh, bias_l2, label="L2 Bias")
ax[0, 0].plot(hh, bias_l1, label="L1 Bias")
ax[0, 0].set_title("FoBG bias wrt ZoBG")
ax[0, 0].legend()
ax[0, 0].set_xlabel("H")

# 2. Plot gradient estiamtes
sns.lineplot(df, x="H", y="zobgs", ax=ax[1, 0], errorbar="sd", label="ZoBGs")
sns.lineplot(df, x="H", y="fobgs", ax=ax[1, 0], errorbar="sd", label="FoBGs")
if zobgs_no_grad is not None:
    sns.lineplot(
        df,
        x="H",
        y="zobgs_no_grad",
        ax=ax[1, 0],
        errorbar="sd",
        label="ZoBGs no grad",
    )
if zobgs_analytical is not None:
    sns.lineplot(
        df,
        x="H",
        y="zobgs_analytical",
        ax=ax[1, 0],
        errorbar="sd",
        label="ZoBGs analytical",
    )
ax[1, 0].set_title("Gradient estimate wrt action")
ax[1, 0].set_ylabel(None)
ax[1, 0].legend()

# 3. Plot variance
var = np.var(zobgs, axis=1)
ax[0, 1].plot(hh, var, label="ZoBGs")
var = np.var(fobgs, axis=1)
ax[0, 1].plot(hh, var, label="FoBGs")
if zobgs_no_grad is not None:
    var = np.var(zobgs_no_grad, axis=1)
    ax[0, 1].plot(hh, var, label="True ZoBGs")
if zobgs_analytical is not None:
    var = np.var(zobgs_analytical, axis=1)
    ax[0, 1].plot(hh, var, label="Analytical ZoBGs")
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
