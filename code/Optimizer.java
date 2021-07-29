import java.util.ArrayList;

/**
 * Interface used by all optimizers that enforces the implementation of the processGradient() function
 */
public interface Optimizer {
    
    /**
     * Processes a raw gradient, possibly performs an adjustment based on an optimizer state,
     * the returns a gradient to be applied to the model's parameters.
     * @param rawGradient The raw, unprocessed gradient for the model evaluated at a data point
     * @return The processed gradient
     */
    public ArrayList<float[][]> processGradient(ArrayList<float[][]> rawGradient);
}