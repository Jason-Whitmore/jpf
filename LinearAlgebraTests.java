import java.util.ArrayList;

public class LinearAlgebraTests{

    public static void initializeConstantTest1(){
        float[] output = LinearAlgebra.initializeConstant(3, 1);
        float[] expected = {1f, 1f, 1f};

        assert Utility.equal(output, expected);
    }


    public static void initializeConstantTest2(){
        float[][] output = LinearAlgebra.initializeConstant(2, 2, 1);

        float[][] expected = new float[2][];
        expected[0] = LinearAlgebra.initializeConstant(2, 1);
        expected[1] = LinearAlgebra.initializeConstant(2, 1);

        assert Utility.equal(expected, output);
    }

    public static void initializeFromArrayListTest(){
        ArrayList<ArrayList<Float>> input = new ArrayList<>();

        input.add(new ArrayList<Float>());
        input.add(new ArrayList<Float>());

        input.get(0).add(1f);
        input.get(0).add(2f);
        input.get(1).add(3f);
        input.get(1).add(4f);
        input.get(1).add(5f);

        float[][] output = LinearAlgebra.initializeFromArrayList(input);
        
        float[][] expected = new float[2][];
        float[] row1 = {1, 2};
        float[] row2 = {3, 4, 5};

        expected[0] = row1;
        expected[1] = row2;

        assert Utility.equal(output, expected);
    }

    public static void initializeIdentityMatrixTest(){
        int input = 3;
        float[][] expected = LinearAlgebra.initializeConstant(input, input, 0);
        for(int i = 0; i < input; i++){
            expected[i][i] = 1;
        }

        float[][] output = LinearAlgebra.initializeIdentityMatrix(input);

        assert Utility.equal(expected, output);
    }

    public static void runTests(){
        LinearAlgebraTests.initializeConstantTest1();
        LinearAlgebraTests.initializeConstantTest2();

        LinearAlgebraTests.initializeFromArrayListTest();
    }

    public static void main(String[] args){
        LinearAlgebraTests.runTests();
    }
}