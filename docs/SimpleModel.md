# SimpleModel

The SimpleModel abstract class defines the template for parameterized functions that have exactly one input and one output vector.

## Notable features

The SimpleModel abstract class was designed to provide features common to single input/output vector parameterized functions. The abstract class provides the fields and methods to access and modify the parameters.

In order to create a derived class, a user must implement the abstract methods in addition to a constructor. Namely, the fit() and predict() methods.

## Derived classes

### Linear Model

The LinearModel class models the simple linear function of f(x) = Wx + b, where W and b are the weight and bias parameters to be learned during training.

This model is used when the data clearly has linear relationships in it.

Due to it's simple nature, the LinearModel's parameter count is directly determined by the input and output vector sizes. This means that the user has no control over the complexity/capacity of the model like the other parameterized models in this library provide.



### Polynomial Model

For more complicated functions, the PolynomialModel class can model nonlinear functions using polynomials for approximation. Expressed mathematically, a component of the output vector is:


Where theta represents the parameters of the function.

Unlike the LinearModel class, the PolynomialModel class allows the user to determine the number of parameters and capacity of the model by adjusting the degree variable in the constructor. A high degree may fit the training data better, but also risks overfitting the dataset.

Even as model complexity increases with the degree of the polynomial, an advantage to using a PolynomialModel to a neural network is that polynomials are much easier to visualize. One only needs to train the model, print out the parameters, and then plot the function with the parameters as coefficients.
