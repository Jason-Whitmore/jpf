# Layer abstract class

The layer abstract class provides the template for all layers in a neural network.

## Fields
All layers have some data in common and are declared in the Layer abstract class fields.

### Fields used for forward propogation passes (predictions)

- Parameters as an ArrayList of float matricies.

- ArrayList of pointers to output layers.

- Input and output vectors.

### Fields used for backward propagation passes (finding gradients)

- ArrayList of pointers to input layers.

- Vector derivative of Loss function with respect to input and output vectors.

- Gradients as an ArrayList of float matricies.


The ArrayList fields can be simply initialized using the basic constructor (or by using super() in subclasses).


## Methods

There are two primary methods that need to be implemented for a Layer class to be functional and can be used with both the fit() and predict() functions of the NeuralNetwork class.

### forwardPass()

The forwardPass() method performs all of the steps used to take in an input vector and produce an output vector. In all layers except for Input layers, the input vector will be populated by using the output vector(s) of previous layers. Once the layer's computations are complete, the result is placed into the output vector of the layer. The process repeats for each layer in the neural network.


### backwardPass()

The backwardPass() method performs all of the steps needed to compute the parameter loss gradient for this layer, as well as set up the vector derivatives for preceding layers. This method performs much more mathematically advanced computations, since multivariate calculus must be properly applied to obtain the proper gradients. This backward pass needs to accompolish two things: Find the gradient of the loss function with respect to the layer parameters, and find the gradient of the loss function with respect to the input vector.


## Derived classes

### Input

The Input layer serves as the starting point of a neural network. Here, the layer accepts the user provided input vector and simply outputs the same vector to the output/next layers. Due to the simple nature of this layer, there are no layer parameters or gradients. The forwardPass() method simply copies the input vector to the output vector. The backwardPass() method simply returns without actions, since there are no gradients to calculate at this layer.

Expressed with linear algebra, the Input layer is quite simple:



