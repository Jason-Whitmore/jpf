# Model

The Model abstract class provides a general definition for any parameterized function. That is, a function whose behavior is dependent
on parameters that can be learned through fitting on training data.

## Features

Since the idea of a parameterized function is vague, this abstract class only provides fields for the parameters themselves (implemented
as an ArrayList of 2d float arrays) and a simple parameter count variable.

The methods provided by the Model abstract class provide the functionality that all parameterized functions have. This includes returning the parameters
(getParameters()), counting the number of parameters (getParameterCount()), and saving the model to disk (saveModel()).

The saveModel() method is an abstract method since all parameterized functions require the ability to save the model to disk, but the details of the
method need to be implemented in a concrete derived class.

## Derived classes

![Model summary UML diagram](images/uml_core.png)

There are two "branches" of classes that are derived from the Model abstract class.

The first branch implements parameterized functions that have one vector as input and one vector as output (via the [SimpleModel](SimpleModel.md) abstract class).
The concrete classes of [LinearModel](LinearModel.md) and [PolynomialModel](PolynomialModel.md) are derived from the SimpleModel abstract class.

The other branch implements parameterized functions that can have as many input and output vectors as needed. So far, this includes the [NeuralNetwork](NeuralNetwork.md) class.