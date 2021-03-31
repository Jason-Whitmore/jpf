


public class Softmax extends Layer{

    private int numUnits;

    public Softmax(Layer inputLayer){
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

        for(int i = 0; i < getOutputLayers().size(); i++){
            for(int j = 0; j < getOutputLayers().get(i).getdLdX().length; j++){
                getdLdY()[j] += getOutputLayers().get(i).getdLdX()[j];
            }
        }

        for(int i = 0; i < this.inputVector.length; i++){

            //Determine the sum of exponents that are not x[i]
            float constant = 0;
            for(int j = 0; j < this.inputVector.length; j++){
                if(j != i){
                    constant += (float)Math.exp((double)this.inputVector[j]);
                }
            }

            double exponential = Math.exp((double)this.inputVector[i]);

            //gradient from the calculus quotient rule
            double gradient = (((exponential + constant) * exponential) - (exponential * exponential)) / (float)Math.pow((double)exponential, 2.0);

            this.dLdX[i] = (float)gradient;
        }

    }
}