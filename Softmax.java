


public class Softmax extends Layer{

    private int numUnits;

    public Softmax(Layer inputLayer){
        super();

        this.numUnits = inputLayer.outputVector.length;

        this.inputVector = new float[this.numUnits];
        this.outputVector = new float[this.numUnits];


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

    }
}