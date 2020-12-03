

public abstract class ActivationFunction{

    public abstract float f(float x);

    public abstract float fPrime(float x);


    public void f(float[] x, float[] dest){
        for(int i = 0; i < x.length; i++){
            dest[i] = f(dest[i]);
        }
    }

    public float[] f(float[] x){
        float[] r = new float[x.length];

        f(x, r);

        return r;
    }



    public void fPrime(float[] x, float[] dest){
        for(int i = 0; i < x.length; i++){
            dest[i] = fPrime(x[i]);
        }
    }

    public float[] fPrime(float[] x){
        float[] r = new float[x.length];

        fPrime(x, r);

        return r;
    }
}