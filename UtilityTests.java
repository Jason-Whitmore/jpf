import java.util.ArrayList;

/**
 * This class contains the code used to perform tests on static functions in the Utility class.
 */
public class UtilityTests{


    private static void clipTest1(){
        float[] array = {-1, 0, 2};

        Utility.clip(array, 0, 1);

        float[] expected = {0,0,1};

        assert Utility.equal(array, expected);
    }

    private static void clipTest2(){
        float[][] array = new float[3][3];
        array[0][0] = -1f;
        array[2][2] = 4f;
        
        Utility.clip(array, 0, 1);

        float[][] expected = new float[3][3];
        expected[2][2] = 1;

        assert Utility.equal(array, expected);
    }

    private static void clipTest3(){
        ArrayList<float[][]> list = new ArrayList<float[][]>();
        list.add(LinearAlgebra.initializeConstant(3, 3, 4));
        list.add(LinearAlgebra.initializeConstant(3, 3, 5));
        list.add(LinearAlgebra.initializeConstant(3, 3, 6));

        Utility.clip(list, 0, 3);

        float[][] expected = LinearAlgebra.initializeConstant(3, 3, 3);
        for(int i = 0; i < list.size(); i++){
            assert Utility.equal(list.get(i), expected);
        }
    }


    private static void sumTest1(){
        float[] input = LinearAlgebra.initializeConstant(5, 1);

        float actualSum = Utility.sum(input);

        assert actualSum == 5;
    }

    private static void sumTest2(){
        float[] input = LinearAlgebra.initializeConstant(10, -2);

        float actualSum = Utility.sum(input);

        assert actualSum == -20;
    }


    private static void meanTest1(){
        float[] input = LinearAlgebra.initializeConstant(10, 3);

        float actualMean = Utility.mean(input);

        assert actualMean == 3f;
    }

    private static void meanTest2(){
        float[] input = {1f,2f,3f};

        float actualMean = Utility.mean(input);

        assert actualMean == 2f;
    }

    private static void meanTest3(){
        float[][] input = LinearAlgebra.initializeConstant(3, 3, 2);

        float actualMean = Utility.mean(input);

        assert actualMean == 2;
    }

    private static void clearArrayTest1(){
        float[] input = LinearAlgebra.initializeConstant(8, 10);

        Utility.clearArray(input);

        float[] expected = LinearAlgebra.initializeConstant(8, 0);

        assert Utility.equal(input, expected);
    }


    private static void clearArrayTest2(){
        float[][] input = LinearAlgebra.initializeConstant(5, 5, 10);

        Utility.clearArray(input);

        float[][] expected = LinearAlgebra.initializeConstant(5, 5, 0);

        assert Utility.equal(input, expected);
    }


    private static void clearArraysTest(){
        ArrayList<float[][]> inputList = new ArrayList<float[][]>();

        for(int i = 1; i <= 10; i++){
            inputList.add(LinearAlgebra.initializeConstant(5, 5, i));
        }

        Utility.clearArrays(inputList);

        float[][] expected = LinearAlgebra.initializeConstant(5, 5, 0);

        for(int i = 0; i < 10; i++){
            assert Utility.equal(inputList.get(i), expected);
        }
    }

    private static void scaleArrayTest1(){
        float[] inputArray = {0, 1, 2};
        float[] inputArrayCopy = {0, 1, 2};
        float inputScalar = 4;

        Utility.scaleArray(inputArray, inputScalar);

        for(int i = 0; i < inputArray.length; i++){
            assert inputArray[i] == inputArrayCopy[i] * inputScalar;
        }
    }


    private static void scaleArrayTest2(){
        float[][] inputArray = LinearAlgebra.initializeConstant(3, 3, 2);
        float inputScalar = 2;

        Utility.scaleArray(inputArray, inputScalar);

        float[][] expectedArray = LinearAlgebra.initializeConstant(3, 3, 4);

        assert Utility.equal(inputArray, expectedArray);
    }


    private static void cloneArraysTest(){
        float[][] a = LinearAlgebra.initializeConstant(3, 3, 0);
        float[][] b = LinearAlgebra.initializeConstant(3, 4, 1);

        ArrayList<float[][]> list = new ArrayList<float[][]>();
        list.add(a);
        list.add(b);

        ArrayList<float[][]> clone = Utility.cloneArrays(list);

        assert Utility.equal(a, clone.get(0));
        assert Utility.equal(b, clone.get(1));
    }

    private static void addArrayTest(){
        float[][] dest = LinearAlgebra.initializeConstant(3, 3, 4);
        float[][] b = LinearAlgebra.initializeConstant(3, 3, 2);
        float scalar = 2;

        float[][] expected = LinearAlgebra.initializeConstant(3, 3, 8);

        Utility.addArray(dest, b, scalar);

        assert Utility.equal(dest, expected);
    }


    private static void addListTest(){
        float[][] a1 = LinearAlgebra.initializeConstant(3, 3, 2);
        float[][] a2 = LinearAlgebra.initializeConstant(4, 4, 3);
        ArrayList<float[][]> listA = new ArrayList<float[][]>();
        listA.add(a1);
        listA.add(a2);

        float[][] b1 = LinearAlgebra.initializeConstant(3, 3, 1);
        float[][] b2 = LinearAlgebra.initializeConstant(4, 4, 2);
        ArrayList<float[][]> listB = new ArrayList<float[][]>();
        listB.add(b1);
        listB.add(b2);

        float scalar = 2f;

        Utility.addList(listA, listB, scalar);

        assert Utility.equal(listA.get(0), LinearAlgebra.initializeConstant(3, 3, 4));
        assert Utility.equal(listA.get(1), LinearAlgebra.initializeConstant(4, 4, 7));
    }






    public static void runTests(){
        UtilityTests.clipTest1();
        UtilityTests.clipTest2();
        UtilityTests.clipTest3();

        UtilityTests.sumTest1();
        UtilityTests.sumTest2();

        UtilityTests.meanTest1();
        UtilityTests.meanTest2();
        UtilityTests.meanTest3();

        UtilityTests.clearArrayTest1();
        UtilityTests.clearArrayTest2();

        UtilityTests.clearArraysTest();

        UtilityTests.scaleArrayTest1();
        UtilityTests.scaleArrayTest2();

        UtilityTests.cloneArraysTest();

        UtilityTests.addArrayTest();
        
        UtilityTests.addListTest();
    }

    public static void main(String[] args){
        UtilityTests.runTests();
    }
}