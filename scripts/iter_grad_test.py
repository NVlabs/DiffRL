from torch import tensor, zeros
from time import time

start = time()
x = tensor([1.0], requires_grad=True)
y = [zeros(1) for _ in range(100)]
y[0] += x**2
for i in range(1, len(y)):
    y[i] += y[i - 1] ** 2
y[-1].backward()
end = time() - start
print("computed real grads in {:.4f}s".format(end))
print("dx", x.grad)

# # now get iterative grads
start = time()
x = tensor([1.0], requires_grad=True)
y = [zeros(1) for _ in range(100)]
y[0] += x**2
y[0].backward(retain_graph=True)
for i in range(1, len(y)):
    y[i] += y[i - 1] ** 2
    x.grad.zero_()  # introduces extra compute but makes gradients correct
    y[i].backward(retain_graph=True)
end = time() - start
print("computed iterative grads in {:.4f}s".format(end))
print("dx", x.grad)
