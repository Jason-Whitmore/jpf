# Extending classes

The design of this library allows for users to extend the classes and create their own classes which fit seamlessly into the existing functionality. Although 
many of the classes of extendable, the classes which make the most sense to extend will be discussed here. It is also highly recommended that the source code for the parent class be examined before extending for additional comments.

## SimpleModel

Extending the SimpleModel class only requires a few implementations of abstract methods before the fully functionality can be realized.
Namely, predict(float[]), calculateGradient(float[], float[], Loss) and saveModel() method. With both of these methods implemented, 
the class will be able to make batch predictions, fit to training data, calculate loss, and save the model to disk with no additional method implementation required from the user.

Although not strictly required, creating a constructor that accepts a filepath to the output file from saveModel() will allow the extended class to create
an instance from a disk saved model.

## Layer


## Loss


## Optimizer