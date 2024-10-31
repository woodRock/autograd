# autograd

Autograd is a forward and reverse mode Automatic Differentiation. Autograd also supports optimization. Automatic differentiation is particularly important in the field of machine learning. For example, it allows one to implement backpropagation in a neural network without a manually-computed derivative

https://en.wikipedia.org/wiki/Automatic_differentiation

## Running 

First compile the code.

```bash
$ g++ tensor.cpp -o output
```

(Optional) compile with multi-threading support.

```bash
$ g++ -fopenmp -O3 tensor.cpp -o output
```

Then run

```bash
$ ./out
```