# JPF: Java Parameterized Functions

## Overview


## Motivation


## Design


## Important components

The following classes and interfaces make up the core of this package's functionality.

- Model (abstract class)
- [SimpleModel](docs/SimpleModel.md) (abstract class)
- [LinearModel](docs/LinearModel.md)
- [PolynomialModel](docs/PolynomialModel.md)
- [NeuralNetwork]()
    - [Layer]() (abstract class)
        - ActivationFunction (interface)
- [Loss](docs/Loss.md)
- [Optimizer](docs/Optimizer.md) (interface)

## Tests

This project demonstrates software testing via black box and unit tests.

The black box tests are used to test correctness of the LinearModel, PolynomialModel, and NeuralNetwork classes.
These black box tests work by conducting several scenarios involving parameterized functions and seeing if their
behavior matches what is expected. For example, when fitting a Neural Network to some training data, it is expected
that the training loss will decrease. These black box tests are implemented as [examples](docs/Examples.md) that have clear goals for
expected outcomes. Check the "Running examples" subsection in the "How to run" section to learn how to run these examples.


Unit testing is done on both the Utility and LinearAlgebra classes. These classes are well suited for unit testing
since they contain very small static helper functions with predictable outputs. The unit test cases are located in
Tests.java file. Check the "Running tests" subsection in the "How to run" section to learn how to run these tests.

## How to run