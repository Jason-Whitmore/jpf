
/**
 * This class contains the code used to perform tests on static functions in the Utility class.
 */
public class UtilityTests{


    private void clipTest1(){
        float[] array = {-1, 0, 2};

        Utility.clip(array, 0, 1);

        float[] expected = {0,0,1};

        assert Utility.equal(array, expected);
    }

    private void clipTest2(){
        float[][] array = new float[3][3];
        array[0][0] = -1f;
        array[2][2] = 4f;
        
        Utility.clip(array, 0, 1);

        float[][] expected = new float[3][3];
        expected[2][2] = 1;

        assert Utility.equal(array, expected);
    }

    public static void main(String[] args){

    }
}