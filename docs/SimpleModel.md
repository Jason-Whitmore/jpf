# SimpleModel

The SimpleModel abstract class defines the template for parameterized functions that have exactly one input and one output vector.

## Notable interface features



## Derived classes

### Linear Model

The LinearModel class models the simple linear function of f(x) = Wx + b, where W and b are the weight and bias parameters to be learned during training.

This model is used when the data clearly has linear relationships in it.

Due to it's simple nature, the LinearModel's parameter count is directly determined by the input and output vector sizes. This means that the user has no control over the complexity/capacity of the model like the other parameterized models in this library provide.



### Polynomial Model


