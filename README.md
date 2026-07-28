# ML-framework
Mini deep learning Python framework inspired by Pytorch and based in C++ I made for learning purposes.

## Structure
The framework works with a series of modules which work together to create an AI model.
Loss functions, activation layers, and the model class themselves have functions for forward
and backward passes. The forward functions are used to get a standard output from an input usually
returning a single value based on the weights and biases in a layer or the operations of a function.

### Ex:
    import dlf

    criterion = dlf.MSEloss()
    t1 = Tensor([2,2], 1)
    t2 = Tensor([2,2], 3)
    loss = criterion.forward(t1, t2)
    print(loss) 
    # Prints [4.000000]

The backwards functions are mainly to be used for model training. They output a Tensor which is inteded to
be passed into the backward function of the module which came before it, The backward and step functions in
the model class handle this chain of backwards passes for a model object. Unlike Pytorch the changing of
weights and biases and handled by the layers themselves rather than an optimizer object.
