import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

filename = "CartPoleSwingUpEnv_grads_ms_200.npz"
print("Loading", filename)

data = np.load(filename)
fobgs = data["fobgs"].squeeze()
zobgs = data["zobgs"].squeeze()
zobgs = np.nan_to_num(zobgs)
zobgs_analytical = (
    data["zobgs_analytical"].squeeze() if "zobgs_analytical" in data else None
)
zobgs_no_grad = data["zobgs_no_grad"].squeeze() if "zobgs_no_grad" in data else None
if zobgs_analytical is not None:
    print("Found analyical!")
if zobgs_no_grad is not None:
    print("Found no grad!")

print("Loaded grads with H={:} and {:} samples".format(zobgs.shape[0], zobgs.shape[1]))

df = pd.DataFrame(columns=["H", "zobgs", "fobgs", "zobgs_no_grad", "zobgs_analytical"])
for i in range(fobgs.shape[0]):
    d = {"H": i + 1, "zobgs": zobgs[i], "fobgs": fobgs[i]}
    if zobgs_no_grad is not None:
        d.update({"zobgs_no_grad": zobgs_no_grad[i]})
    if zobgs_analytical is not None:
        d.update({"zobgs_analytical": zobgs_analytical[i]})
    df = pd.concat((df, pd.DataFrame(d)))
df = df.explode(["zobgs", "fobgs", "zobgs_no_grad", "zobgs_analytical"])
df = df.reset_index()

print("Plotting")
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

sns.lineplot(df, x="H", y="zobgs", ax=ax1, label="ZoBGs")
sns.lineplot(df, x="H", y="fobgs", ax=ax1, label="FoBGs")
if zobgs_no_grad is not None:
    sns.lineplot(df, x="H", y="zobgs_no_grad", ax=ax1, label="ZoBGs no grad")
if zobgs_analytical is not None:
    sns.lineplot(df, x="H", y="zobgs_analytical", ax=ax1, label="ZoBGs analytical")
ax1.set_title("Gradient estimate wrt action")
ax1.set_ylabel(None)
ax1.legend()

var = np.var(zobgs, axis=1)
ax2.plot(np.arange(len(var)) + 1, var)
var = np.var(fobgs, axis=1)
ax2.plot(np.arange(len(var)) + 1, var)
if zobgs_no_grad is not None:
    var = np.var(zobgs_no_grad, axis=1)
    ax2.plot(np.arange(len(var)) + 1, var)
if zobgs_analytical is not None:
    var = np.var(zobgs_analytical, axis=1)
    ax2.plot(np.arange(len(var)) + 1, var)
ax2.set_yscale("log")
ax2.set_xlabel("H")
ax2.set_title("Gradient variance")
plt.tight_layout()

filename = filename.split(".")[0] + ".pdf"
print("Saving to {:}".format(filename))
plt.savefig(filename)
