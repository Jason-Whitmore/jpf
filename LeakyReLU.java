


public class LeakyReLU extends ActivationFunction{

    private float alpha;

    public LeakyReLU(float alpha){
        this.alpha = alpha;
    }

    public float f(float x){
        if(x > 0){
            return x;
        } else {
            return x * alpha;
        }
    }

    public float fPrime(float x){
        if(x > 0){
            return 1;
        } else {
            return alpha;
        }
    }


    public String toString(){
        return "LEAKYRELU(" + this.alpha + ")";
    }
}