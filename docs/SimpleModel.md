# SimpleModel

The SimpleModel abstract class defines the template for parameterized functions that have exactly one input and one output vector.

## Features

The SimpleModel abstract class was designed to provide features common to single input/output vector parameterized functions. Mathematically, SimpleModels can be expressed as:

![SimpleModel equation](images/simplemodel_equation.png)

The abstract class provides the fields and methods to access and modify the parameters as well as training and prediction.

In order to create a derived class, a user must implement the abstract methods in addition to a constructor. Namely, the fit() and predict() methods.

## Derived classes

There are currently two derived classes of SimpleModel: LinearModel and PolynomialModel.


### Polynomial Model

For more complicated functions, the PolynomialModel class can model nonlinear functions using polynomials for approximation. Expressed mathematically, a component of the output vector is:


Where theta represents the parameters of the function.

Unlike the LinearModel class, the PolynomialModel class allows the user to determine the number of parameters and capacity of the model by adjusting the degree variable in the constructor. A high degree may fit the training data better, but also risks overfitting the dataset.

Even as model complexity increases with the degree of the polynomial, an advantage to using a PolynomialModel to a neural network is that polynomials are much easier to visualize. One only needs to train the model, print out the parameters, and then plot the function with the parameters as coefficients.
