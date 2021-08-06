# Extending classes

The design of this library allows for users to extend the classes and create their own classes which fit seamlessly into the existing functionality. Although 
many of the classes of extendable, the classes which make the most sense to extend will be discussed here. It is also highly recommended that the source code for the parent class be examined before extending for additional comments.

## SimpleModel

Extending the [SimpleModel](SimpleModel.md) class only requires a few implementations of abstract methods before the fully functionality can be realized.
Namely, predict(float[]), calculateGradient(float[], float[], Loss) and saveModel() method. With both of these methods implemented, 
the class will be able to make batch predictions, fit to training data, calculate loss, and save the model to disk with no additional method implementation required from the user.

Although not strictly required, creating a constructor that accepts a filepath to the output file from saveModel() will allow the extended class to create
an instance from a disk saved model.

## Layer

Extending the [Layer](Layer.md) class mostly involves implementing the forwardPass() and backwardPass() methods which are used for prediction and gradient calculation, respectively. 

The forwardPass() method is responsible for populating its own input vector, typically by "pulling" the output vectors from the Layer's input layers. If the Layer is restricted to one input layer, then typically the input vector is just copied from the input layer's output vector. Once this is accompolished, some computation should be done on the input vector (likely using any layer parameters) and the result should be placed in the output vector field.

The backwardPass() method is responsible for populating its own dLdY vector, typically by "pulling" the dLdX vectors from the Layer's output layers. This is done by adding together all of the dLdX vectors using the initializedlDY() method at the start of the backwardPass() implementation. Then, this dLdY vector is used to calculate the gradient of the loss function with respect to the parameters. This process can be fairly calculus heavy and typically involves using the chain rule and the populated input vector. One the gradients are populated inside of the gradient list, the dLdX vector needs to be calculated to before the backwardPass() implementation is complete.

Although not needed for prediction and fitting, the toString() method should also be implemented as well as a string constructor so that layers can be constructed from a string representation when the neural network is saved to disk. The static createLayerFromString() method should also be modified to create the new extended class.

## Loss

Implementing the [Loss](Loss.md) interface involves completing the implementation of three methods.

The calculateLossVector(float[], float[]) method calculates the loss function value at each component of the predicted and labeled output vectors. It is noteworthy that this method is not the single variate loss function that is typically used. Instead, it is recommended to implement the single variate loss function separately and then call that method inside of the calculateLossVector(float[], float[]) method.

The calculateLossVectorGradient(float[], float[]) method calculates the loss function gradient at each component of the predicted and labeled output vectors. This involves getting dLdY for each component in the Y, or output, vector. It's important that dLdY be implemented here (steepest ascent of loss function) rather than -dLdY (steepest descent of loss function) since the flipping of the sign will be done during the fitting process.

The calculateLossScalar(float[], float[]) method calculates the scalar value of the loss function for the predicted and labeled output vectors. This method is not used during the fitting or prediction process and is instead used to display the number to the user. Typically this is implemented as a mean of the vector components produced from calculateLossVector(float[], float[]) although ultimately it's up to the user to choose how to calculate the scalar.

## Optimizer

Implementing the [Optimizer](Optimizer.md) interface involves completing the implementation of only one method.

The processGradient(ArrayList<float[][]>) method will take in an unprocessed gradient calculated from the calculateGradient() method, perform some processing, and then return a new gradient with the exact same dimensions as the input one. The processing performed on the gradients depends on what the user's goal is. For example, the RMSProp optimizer maintains an average learning rate across each parameter.