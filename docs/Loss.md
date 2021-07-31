# Losses

The loss function interface provides the basic template for implementing loss functions that are used to evaluate model performance and fit the model to training data.

## The loss interface

### float[] calculateLossVector(float[] yTrue, float[] yPredicted)

Calculates the loss for each component of the output vector. Typically, this is done elementwise, but isn't required.

### float[] calculateLossVectorGradient(float[] yTrue, float[] yPredicted)

Calculates the gradient of steepest ascent of the loss function for each component of the output vector. In other words, each component is the derivative of the loss function with respect to the output

## Derived classes

### Mean Square Error (MSE)

L(yTrue, yPred) = (yPred - yTrue)^2

The standard loss function for regression tasks. This function is popular because the loss signal becomes very strong as the difference between yPred and yTrue grows larger. Also, this function is twice differentiable.

### Mean Absolute Error (MAE)

L(yTrue, yPred) = |yPred - yTrue|

Used for regression tasks. Less popular than MSE because the loss signal isn't quite as strong as the difference between yPred and yTrue grows larger. Also, the function is not smooth and twice differentiable like MSE is. Still, the output from this function is much more intuitive to directly understand than MSE is.

### Binary Cross Entropy

NOT YET IMPLEMENTED
