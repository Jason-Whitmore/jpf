


public class SoftmaxLayer extends Layer{

    private int numUnits;

    public SoftmaxLayer(Layer inputLayer){
        super();

        this.numUnits = inputLayer.outputVector.length;

        this.inputVector = new float[this.numUnits];
        this.outputVector = new float[this.numUnits];

        this.dLdX = new float[this.numUnits];
        this.dLdY = new float[this.numUnits];

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
}