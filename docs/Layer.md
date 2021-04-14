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
