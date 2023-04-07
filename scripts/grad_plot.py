import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

filename = "CartPoleSwingUpEnv_grads_200.npz"

data = np.load(filename)
fobgs = data["fobgs"].squeeze()
zobgs = data["zobgs"].squeeze()
zobgs = np.nan_to_num(zobgs)

print(
    "Loaded grads with max H={:} and {:} samples".format(zobgs.shape[0], zobgs.shape[1])
)

df = pd.DataFrame(columns=["H", "zobg", "fobg"])
for i in range(fobgs.shape[0]):
    new = pd.DataFrame({"H": i + 1, "zobg": zobgs[i], "fobg": fobgs[i]})
    df = pd.concat((df, new))
df = df.explode(["zobg", "fobg"])
df = df.reset_index()

print("Plotting")
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

sns.lineplot(df, x="H", y="zobg", ax=ax1, label="ZoBGs")
sns.lineplot(df, x="H", y="fobg", ax=ax1, label="FoBGs")
ax1.set_title("Gradient estimate wrt action")
ax1.set_ylabel(None)
ax1.legend()

var = np.var(zobgs, axis=1)
ax2.plot(np.arange(len(var)) + 1, var)
var = np.var(fobgs, axis=1)
ax2.plot(np.arange(len(var)) + 1, var)
ax2.set_yscale("log")
ax2.set_xlabel("H")
ax2.set_title("Gradient variance")
plt.tight_layout()

filename = filename.split(".")[0] + ".pdf"
print("Saving to {:}".format(filename))
plt.savefig(filename)
