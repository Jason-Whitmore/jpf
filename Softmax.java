


class Softmax extends ActivationFunction{

    public Softmax(){

    }

    @Override
    public float[] f(float[] x){

        float[] r = new float[x.length];

        float expSum = 0;
        float[] expVector = new float[x.length];

        for(int i = 0; i < x.length; i++){
            expSum += (float)Math.exp((double)x[i]);

            expVector[i] = (float)Math.exp((double)x[i]);
        }

        for(int i = 0; i < x.length; i++){
            r[i] = expVector[i] / expSum;
        }

        return x;
    }
}