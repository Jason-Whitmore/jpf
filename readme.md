# JPF: Java Parameterized Functions

## Overview

JPF utilizes the Java object oriented programming paradigm to implement parameterized functions which allow users the ability to create, fit, evaluate, save, and load machine learning models using native Java. 


Project start date: October 3rd, 2020

## Motivation

I was motivated to create this project for several reasons.

First, I was intrigued by the mathematics behind supervised machine learning and figured that the best way to learn them would be to understand the 
multivariate calculus behind it and then implement the math myself.

Second, I wanted to focus on my Java programming skills and decided that this project would provide an excellent opportunity to use Java's object oriented
design features to build a software package that would contain components that were easily testable, extendable, and understandable. The Tensorflow and Keras
machine learning libraries that are widely used in Python are important inspirations for this project. These libraries made prototyping neural network and machine learning techniques easy, fast, and intuitive.

## Design

The JPF library is designed using Java's OOP and abstraction mechanisms. Use of abstract classes and interfaces allows for fast development of classes
through code reuse and inheritance of data and behavior. The complete UML summary design is available [here](docs/images/uml_all.png).
The Model class hierarchy, which is a subset of the complete package design, demonstrates these class relationships:

![Model UML diagram](docs/images/uml_core.png)

More sections of this package's design are discussed in the various pages of documentation.

This design also makes it easier for users to extend classes to create their own components that seamlessly integrate with this library. For example, following the
documentation for extending classes, a user can create their own layers, loss functions, optimizers, and activation functions.

## Documentation

- [Model](docs/Model.md) (abstract class)
    - [SimpleModel](docs/SimpleModel.md) (abstract class)
        - [LinearModel](docs/LinearModel.md)
        - [PolynomialModel](docs/PolynomialModel.md)
    - [NeuralNetwork](docs/NeuralNetwork.md)
        - [Layer](docs/Layer.md) (abstract class)
            - [ActivationFunction](docs/ActivationFunctions.md) (abstract class)
- [Loss](docs/Loss.md) (interface)
- [Optimizer](docs/Optimizer.md) (interface)
- [Examples](docs/Examples.md)
- [Miscellaneous classes](docs/Misc.md)
- [Extending classes](docs/ExtendingClasses.md)

## Tests

This project demonstrates software testing via black box and unit tests.

The black box tests are used to test correctness of the LinearModel, PolynomialModel, and NeuralNetwork classes.
These black box tests work by conducting several scenarios involving parameterized functions and seeing if their
behavior matches what is expected. For example, when fitting a Neural Network to some training data, it is expected
that the training loss will decrease. These black box tests are implemented as [examples](docs/Examples.md) that have clear goals for
expected outcomes. Check the "Examples" subsection in the "How to use/run" section to learn how to run these examples.


Unit testing is done on both the Utility and LinearAlgebra classes. These classes are well suited for unit testing
since they contain very small static helper functions with predictable outputs. The unit test cases are located in
Tests.java file. Check the "Tests" subsection in the "How to use/run" section to learn how to run these tests.

## How to use/run

### Creating the jpf.jar file

A jar file containing the JPF package can be created by using the following make command in the main directory:

```
make jpf.jar
```

This will compile all of the package source files and place the resulting class files into a jar file. This jar file can
then be moved into a separate project directory, where the package can then be imported as "import jpf.*;".

### Creating Java documentation

Javadocs can be created using the following make command:

```
make javadocs
```

This will create a folder called "javadocs" which contains the javadocs as html files. Of particular interest is the index.html file, which is a good starting point to
explore all of the javadocs.

### Examples

The [examples](docs/Examples.md) documentation describes what each specific example does and display sample output. To run, simply run the following commands in the main directory:

```
make examples
java Examples
```

The "make examples" command will run a script to create the jpf.jar if needed, then compile the Examples.java source code alongside the jpf.jar package code.

The "java Examples" command will run the Examples program and print out the various command line arguments that can be used to run the examples.

### Tests

To run the unit tests for the Utility.java and LinearAlgebra.java classes, simply run the following commands in the main directory:

```
make tests
java -ea Tests
```

The "make tests" command will run a script to create the jpf.jar if needed, then compile the Tests.java source code alongside the jpf.jar package code.

The "java -ea Tests" command will run the tests with assert statements enabled, which will warn the user if a test case failed.