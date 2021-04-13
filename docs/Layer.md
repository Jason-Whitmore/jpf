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
