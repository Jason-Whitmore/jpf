/**
 * Class that defines the SoftmaxLayer class which is derived from the Layer abstract class.
 * This layer is used when the output needs to be a discrete probability distribution.
 */
public class SoftmaxLayer extends Layer{

    /**
     * Basic constructor for the SoftmaxLayer, which output a discrete probability distribution.
     * @param inputLayer The layer whose output vector is the input vector for this layer.
     * It is highly recommended that the input layer be a Dense layer with a linear activation function.
     */
    public SoftmaxLayer(Layer inputLayer){
        super();
        Utility.checkNotNull(inputLayer);

        int numUnits = inputLayer.outputVector.length;

        this.inputVector = new float[numUnits];
        this.outputVector = new float[numUnits];

        this.dLdX = new float[numUnits];
        this.dLdY = new float[numUnits];

    }

    /**
     * Constructs a SoftmaxLayer from a string output by this layer's toString() method.
     * Example: SOFTMAX(5) is a softmax layer of size 5.
     * @param layerInfoString
     */
    public SoftmaxLayer(String layerInfoString){
        super();
        Utility.checkNotNull(layerInfoString);

        String numUnitsString = layerInfoString.replace("SOFTMAX(", "").replace(")", "");
        int numUnits = Integer.parseInt(numUnitsString);

        if(numUnits <= 0){
            throw new AssertionError("numUnits should be >= 0");
        }

        this.inputVector = new float[numUnits];
        this.outputVector = new float[numUnits];

        this.dLdX = new float[numUnits];
        this.dLdY = new float[numUnits];
    }


    public void forwardPass(){

        float expSum = 0;

        for(int i = 0; i < this.inputVector.length; i++){
            expSum += (float)Math.exp((double)this.inputVector[i]);
        }

        for(int i = 0; i < this.inputVector.length; i++){
            this.outputVector[i] = ((float)Math.exp((float) this.inputVector[i])) / expSum;
        }
    }

    public void backwardPass(){
        //Determine error vector from next layers
        initializedLdY();

        //Determine sum of exponents before doing the backprop step
        float constant = 0;
        for(int i = 0; i < this.inputVector.length; i++){
            constant += (float)Math.exp((double)this.inputVector[i]);
        }

        for(int i = 0; i < this.inputVector.length; i++){

            //Subtract the current exponential from the constant
            constant -= (float)Math.exp((double) this.inputVector[i]);

            double exponential = Math.exp((double)this.inputVector[i]);

            //gradient from the calculus quotient rule
            double gradient = (((exponential + constant) * exponential) - (exponential * exponential)) / (float)Math.pow((double)exponential, 2.0);

            this.dLdX[i] = (float)gradient;

            //Add the current exponential to the constant
            constant += (float)Math.exp((double) this.inputVector[i]);
        }

    }


    public String toString(){
        int numUnits = this.outputVector.length;
        return "Softmax(" + numUnits + ")";
    }
}