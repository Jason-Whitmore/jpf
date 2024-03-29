# PolynomialModel

The PolynomialModel class is a derived class of the [SimpleModel](SimpleModel.md) abstract class that allows for the user to adjust the capacity/parameters of the model by altering the degree of the underlying polynomial functions.


## Features

The PolynomialModel class can learn more complex non-linear relationships between training input and outputs. This is achieved by modeling each output component as a sum of polynomials, with each polynomial using a component of the input vector as the polynomial function input.

Mathematically, this can be expressed as:

![Polynomial equation](images/polynomial_equation.png)

where y_j is the *j* th output component, D is the model degree, N is the input vector size, and b_j, theta_i,j,k are the model parameters.


## Implementation

PolynomialModel is implemented using the model parameters as coefficients to the underlying polynomial functions.

A specific coefficient can be found using the following convention:

parameters.get(i)[j][k] refers to the coefficient multiplied by the *i* th input component raised to the *(k + 1)* th power which is added to the *j* th output component.

## Examples

There are two [examples](Example.md) in the Examples.java class which demonstrate and test the features of the class: polynomialsin and polynomialoverfit.

## Considerations

Compared to the [LinearModel](LinearModel.md), the PolynomialModel is able to learn much more complex non-linear functions. Additionally, the user is able to choose the capacity of the model by using the degree parameter upon model construction. A high degree will add more parameters to the model, which may help in learning more complex functions, but may also lead to overfitting. Conversely, a low degree PolynomialModel may have difficulty fitting to training data and may also lead to underfitting.
