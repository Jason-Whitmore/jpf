package jpf;

/**
 * Class that defines the SoftmaxLayer class which is derived from the Layer abstract class.
 * This layer is used when the output needs to be a discrete probability distribution.
 */
public class SoftmaxLayer extends Layer{

    private float epsilon;

    /**
     * Basic constructor for the SoftmaxLayer, which output a discrete probability distribution.
     * @param inputLayer The layer whose output vector is the input vector for this layer.
     * It is highly recommended that the input layer be a Dense layer with a linear activation function.
     */
    public SoftmaxLayer(Layer inputLayer){
        super();
        Utility.checkNotNull(inputLayer);

        int numUnits = inputLayer.outputVector.length;
        this.inputLayers.add(inputLayer);

        //Connect input layer to this layer
        this.connectInputAndOutputLayers();

        this.inputVector = new float[numUnits];
        this.outputVector = new float[numUnits];

        this.dLdX = new float[numUnits];
        this.dLdY = new float[numUnits];

        this.epsilon = 0.0001f;
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
            throw new AssertionError("numUnits should be > 0");
        }

        this.inputVector = new float[numUnits];
        this.outputVector = new float[numUnits];

        this.dLdX = new float[numUnits];
        this.dLdY = new float[numUnits];

        this.epsilon = 0.0001f;
    }


    public void forwardPass(){
        //Bring over the previous layer's output into this layer's input vector
        this.initializeInputVectorCopy();

        float expSum = 0;

        for(int i = 0; i < this.inputVector.length; i++){
            expSum += (float)Math.exp((double)this.inputVector[i]);
        }

        expSum += this.epsilon;

        for(int i = 0; i < this.inputVector.length; i++){
            this.outputVector[i] = ((float)Math.exp((float) this.inputVector[i])) / expSum;
        }
    }

    public void backwardPass(){
        //Determine error vector from next layers
        this.initializedLdY();

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
            double dYdX = (constant * exponential) / (float)Math.pow((double)(constant + exponential), 2.0);

            this.dLdX[i] = ((float)dYdX) * this.dLdY[i];
            //System.out.println(this.dLdX[i]);

            //Add the current exponential to the constant
            constant += (float)Math.exp((double) this.inputVector[i]);
        }

    }


    public String toString(){
        int numUnits = this.outputVector.length;
        return "SOFTMAX(" + numUnits + ")";
    }
}