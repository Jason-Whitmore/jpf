# Loss

The loss function interface provides the basic template for implementing loss functions that are used to evaluate model performance and fit the model to training data.

## The loss interface

### float[] calculateLossVector(float[] yTrue, float[] yPredicted)

Calculates the loss for each component of the output vector. Typically, this is done elementwise, but isn't required.

### float[] calculateLossVectorGradient(float[] yTrue, float[] yPredicted)

Calculates the gradient of steepest ascent of the loss function for each component of the output vector. In other words, each component is the derivative of the loss function with respect to the output


### float calculateLossScalar(float[] yTrue, float[] yPredicted)

Calculates the scalar loss on a yTrue vector and a yPredicted vector. Typically this is just the mean of all of the loss components.

## Derived classes

### MSE (Mean Squared Error)

L(yTrue, yPredicted) = (yPredicted - yTrue)^2

The standard loss function for regression tasks. This function is popular because the loss signal becomes very strong as the difference between yPred and yTrue grows larger. Also, this function is twice differentiable.

### CrossEntropy

L(yTrue, yPredicted) =

if yTrue = 1: -log(yPredicted)

if yTrue = 0: -log(1 - yPredicted)

A loss function that can be used for classification tasks. The primary consideration with this loss function is that the labels, yTrue, must be either 0 or 1.
Additionally, the predictied outputs, yPred, must be in range (0, 1). Due to these constraints, this loss function is typically used when the output layers are
either SoftmaxLayers or Dense with a sigmoid activation function.
