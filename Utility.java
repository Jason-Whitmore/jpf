import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Class for assorted static functions that may be useful elsewhere in the project or for the user.
 */
public class Utility{

    /**
     * Calculates randomized indicies and groups them into minibatches
     * @param n The total number of indicies forming the set {0,1,2, ..., n - 1}
     * @param minibatchSize The maximum size of minibatches, or groups, that these indicies are formed into
     * @return The randomized indicies grouped into minibatches
     */
    public static ArrayList<ArrayList<Integer>> getMinibatchIndicies(int n, int minibatchSize){

        if(n < 1){
            throw new AssertionError("n (number of indicies) needs to be >= 1");
        }

        if(minibatchSize < 1){
            throw new AssertionError("Minibatch size needs to be >= 1");
        }

        //get the indicies
        ArrayList<Integer> indicies = new ArrayList<Integer>(n);

        for(int i = 0; i < n; i++){
            indicies.add(i);
        }

        //randomly shuffle the indicies
        Collections.shuffle(indicies);

        //Separate the indicies into minibatches
        ArrayList<ArrayList<Integer>> r = new ArrayList<ArrayList<Integer>>();

        ArrayList<Integer> minibatchIndicies = new ArrayList<Integer>();
        for(int i = 0; i < indicies.size(); i++){

            if(minibatchIndicies.size() >= minibatchSize){
                r.add(new ArrayList<Integer>(minibatchIndicies));
                minibatchIndicies = new ArrayList<Integer>();
            } else {
                minibatchIndicies.add(indicies.get(i));
            }
        }


        if(minibatchIndicies.size() > 0){
            r.add(new ArrayList<Integer>(minibatchIndicies));
        }

        return r;
    }

    /**
     * Clips the values in the array to be in range (min, max)
     * @param array The array to modify.
     * @param min The minimum value an element can be.
     * @param max The maximum value an element can be.
     */
    public static void clip(float[] array, float min, float max){
        Utility.checkNotNull((Object)array);

        if(max < min){
            throw new AssertionError("max parameter is less than min parameter.");
        }

        for(int i = 0; i < array.length; i++){
            if(array[i] < min){
                array[i] = min;
            } else if(array[i] > max){
                array[i] = max;
            }
        }
    }

    /**
     * Clips the values in the 2d array to be in range (min, max)
     * @param array The array to modify.
     * @param min The minimum value an element can be.
     * @param max The maximum value an element can be.
     */
    public static void clip(float[][] array, float min, float max){
        Utility.checkNotNull((Object)array);

        if(max < min){
            throw new AssertionError("max parameter is less than min parameter.");
        }

        for(int i = 0; i < array.length; i++){
            Utility.checkNotNull((Object)(array[i]));
            clip(array[i], min, max);
        }
    }

    /**
     * Clips the values in the collection of 2d arrays to be in range (min, max)
     * @param arrays The collection of 2d arrays to modify
     * @param min The minimum value an element can be.
     * @param max The maximum value an element can be.
     */
    public static void clip(ArrayList<float[][]> arrays, float min, float max){
        Utility.checkNotNull(arrays);

        if(max < min){
            throw new AssertionError("max parameter is less than min parameter.");
        }

        for(int i = 0; i < arrays.size(); i++){
            Utility.checkNotNull((Object)(arrays.get(i)));
            clip(arrays.get(i), min, max);
        }
    }

    /**
     * Computes the sum of the provided data.
     * @param a The input data to sum.
     * @return The sum.
     */
    public static float sum(float[] a){
        Utility.checkNotNull(a);

        float sum = 0;
        
        for(int i = 0; i < a.length; i++){
            sum += a[i];
        }

        return sum;
    }

    /**
     * Computes the mean value of the provided data.
     * @param a The input data to find the mean value of.
     * @return The mean value.
     */
    public static float mean(float[] a){
        Utility.checkNotNull(a);

        return sum(a) / a.length;
    }

    /**
     * Computes the mean value of the provided data.
     * @param a The input data to find the mean value of.
     * @return The mean value.
     */
    public static float mean(float[][] a){
        Utility.checkNotNull((Object)a);

        float sum = 0;
        int entryCount = 0;
        for(int i = 0; i < a.length; i++){
            sum += mean(a[i]);
            entryCount += a.length;
        }

        return sum / entryCount;
    }


    /**
     * Sets all the elements of the array to zero.
     * @param a The array to modify.
     */
    public static void clearArray(float[] a){
        Utility.checkNotNull((Object)a);

        for(int i = 0; i < a.length; i++){
            a[i] = 0f;
        }
    }

    /**
     * Sets all the elements of the 2d array to zero.
     * @param a The 2d array to modify.
     */
    public static void clearArray(float[][] a){
        Utility.checkNotNull((Object)a);

        for(int i = 0; i < a.length; i++){
            Utility.checkNotNull((Object)(a[i]));
            clearArray(a[i]);
        }
    }

    /**
     * Clears all entries in the arrays.
     * @param arrays The list of arrays to clear.
     */
    public static void clearArrays(ArrayList<float[][]> arrays){
        Utility.checkNotNull((Object)arrays);

        for(int i = 0; i < arrays.size(); i++){
            Utility.checkNotNull((Object)(arrays.get(i)));
            clearArray(arrays.get(i));
        }
    }

    /**
     * Scales all elements in the array.
     * @param a The array to modify
     * @param scalar The scaling factor
     */
    public static void scaleArray(float[] a, float scalar){
        Utility.checkNotNull((Object)a);

        for(int i = 0; i < a.length; i++){
            a[i] *= scalar;
        }
    }

    /**
     * Scales all the elements in the 2d array.
     * @param a The 2d array to modify.
     * @param scalar The scaling factor
     */
    public static void scaleArray(float[][] a, float scalar){
        Utility.checkNotNull((Object)a);

        for(int i = 0; i < a.length; i++){
            scaleArray(a[i], scalar);
        }
    }

    /**
     * Clones the input collection of arrays
     * @param arrays The collection of arrays to clone
     * @return A new ArrayList of 2d arrays identical in contents to the input.
     */
    public static ArrayList<float[][]> cloneArrays(ArrayList<float[][]> arrays){
        Utility.checkNotNull(arrays);

        ArrayList<float[][]> ret = new ArrayList<float[][]>(arrays.size());

        for(int i = 0; i < arrays.size(); i++){
            Utility.checkNotNull((Object)(arrays.get(i)));

            int numRows = arrays.get(i).length;
            int numCols = arrays.get(i)[0].length;

            ret.add(new float[numRows][numCols]);

            for(int r = 0; r < numRows; r++){
                Utility.checkNotNull(arrays.get(i)[r]);
                for(int c = 0; c < numCols; c++){
                    ret.get(i)[r][c] = arrays.get(i)[r][c];
                }
            }
        }


        return ret;
    }

    /**
     * Adds one scaled 2d array to another. Used to apply a gradient to a model's parameters.
     * @param dest The first 2d array, which is also where the result is stored.
     * @param b The second 2d array, which is scaled.
     * @param scalar The scalar to multiply the b array by.
     */
    public static void addArray(float[][] dest, float[][] b, float scalar){
        Utility.checkNotNull(dest, b);

        for(int r = 0; r < dest.length; r++){
            Utility.checkArrayLengthsEqual(dest[r], b[r]);

            for(int c = 0; c < dest[r].length; c++){
                dest[r][c] = dest[r][c] + (b[r][c] * scalar);
            }
        }
    }

    /**
     * Adds the one scaled list to another list.
     * @param list The list to add to. This is the destination list.
     * @param newList The scaled list to add to the first parameter.
     * @param scalar The scalar to multiply newList by.
     */
    public static void addList(ArrayList<float[][]> list, ArrayList<float[][]> newList, float scalar){
        Utility.checkNotNull(list, newList);
        Utility.checkListDimensionsEqual(list, newList);

        for(int i = 0; i < list.size(); i++){
            addArray(list.get(i), newList.get(i), scalar);
        }
    }

    /**
     * Scales the list elements by a given factor.
     * @param list The list to scale.
     * @param scalar The factor to scale the list elements by by.
     */
    public static void scaleList(ArrayList<float[][]> list, float scalar){
        Utility.checkNotNull(list);

        for(int i = 0; i < list.size(); i++){
            Utility.checkNotNull((Object)(list.get(i)));

            scaleArray(list.get(i), scalar);
        }
    }

    /**
     * Creates a formatted string of the array contents.
     * @param array The array to turn into a string.
     * @return The formatted string
     */
    public static String arrayToString(float[] array){
        Utility.checkNotNull((Object)array);

        if(array.length == 0){
            return "[]";
        }

        StringBuilder sb = new StringBuilder();

        sb.append("[");

        for(int i = 0; i < array.length - 1; i++){
            sb.append(array[i]);
            sb.append(" ");
        }

        sb.append(array[array.length - 1]);
        sb.append("]");

        return new String(sb);
    }

    /**
     * Creates a formatted string of the 2d array contents.
     * @param array The 2d array to turn into a string.
     * @return The formatted string.
     */
    public static String arrayToString(float[][] array){
        Utility.checkNotNull((Object)array);

        StringBuilder sb = new StringBuilder();

        sb.append("[");
        //sb.append(arrayToString(array[0]));

        for(int i = 0; i < array.length - 1; i++){
            sb.append(arrayToString(array[i]));
            sb.append("\n");
        }

        sb.append(arrayToString(array[array.length - 1]));

        sb.append("]");

        return new String(sb);
    }

    /**
     * Creates a formatted string of the collection of 2d arrays.
     * @param arrays The collection of 2d arrays.
     * @return The formatted string.
     */
    public static String arraysToString(ArrayList<float[][]> arrays){
        Utility.checkNotNull(arrays);

        if(arrays.size() == 0){
            return "[]";
        }
        StringBuilder sb = new StringBuilder();

        sb.append("[\n");

        for(int i = 0; i < arrays.size() - 1; i++){
            sb.append(arrayToString(arrays.get(i)));
            sb.append(",\n");
        }

        sb.append(arrayToString(arrays.get(arrays.size() - 1)));
        sb.append("\n]");

        return new String(sb);
    }

    /**
     * Writes the contents to the filepath.
     * @param filePath The filepath to write to.
     * @param contents The text file contents that will be placed at the filepath.
     * @return True on success, false on failure and prints error message to standard error.
     */
    public static boolean writeStringToFile(String filePath, String contents){
        Utility.checkNotNull(filePath, contents);

        try{
            FileWriter f = new FileWriter(filePath);
            f.write(contents);
            f.close();

        } catch(Exception e){
            System.err.println(e.getMessage());
            return false;
        }

        return true;
    }


    /**
     * Allocates and initializes an array from the string.
     * This function serves as a shortcut for the same function in the LinearAlgebra class.
     * 
     * Example for syntax: [0,1,2,3]
     * @param s The string to create the array from.
     * @return The allocated and initialized array.
     */
    public static float[] stringToArray(String s){
        Utility.checkNotNull(s);
        return LinearAlgebra.stringToArray(s);
    }


    /**
     * Initializes a matrix from a string.
     * This function serves as a shortcut for the same function in the LinearAlgebra class.
     * 
     * Example format:
     * [[1,0,0]
     *  [0,1,0]
     *  [0,0,1]]
     * @param s The string containing the data for a matrix.
     * @return The newly allocated matrix from string data.
     */
    public static float[][] stringToMatrix(String s){
        Utility.checkNotNull(s);
        return LinearAlgebra.stringToMatrix(s);
    }


    /**
     * Converts string representation of multiple matricies into an arraylist of initialized matricies.
     * This function serves as a shortcut for the same function in the LinearAlgebra class.
     * 
     * Example format:
     * 
     * [
     * [[1,2,3]
     * [4,5,6]
     * [7,8,9]],
     * [[1,2,3]
     * [4,5,6]
     * [7,8,9]]
     * ]
     * @param s The string to parse into an arraylist of matricies.
     * @return The initialized arraylist of matricies
     */
    public static ArrayList<float[][]> stringToMatrixList(String s){
        Utility.checkNotNull(s);
        return LinearAlgebra.stringToMatrixList(s);
    }


    /**
     * Gets a random sample from a uniform distribution in [lowerBound, upperBound)
     * @param lowerBound The smallest value that can be returned.
     * @param upperBound The exlusive upper boundary for values that can be returned.
     * @return The random sample.
     */
    public static float getRandomUniform(float lowerBound, float upperBound){
        if(upperBound < lowerBound){
            throw new AssertionError("upperBound is lower than lowerBound");
        }

        float delta = upperBound - lowerBound;

        return lowerBound + delta * ((float)Math.random());
    }

    /**
     * Creates an array from a uniform distribution in [lowerBound, upperBound)
     * @param lowerBound The smallest value that an entry can be.
     * @param upperBound The largest value that an entry can be.
     * @param arrayLength The length of the random array. Should be >= 1
     * @return The randomized array.
     */
    public static float[] getRandomUniform(float lowerBound, float upperBound, int arrayLength){
        if(upperBound < lowerBound){
            throw new AssertionError("upperBound is lower than lowerBound");
        }

        if(arrayLength <= 0){
            throw new AssertionError("arrayLength needs to be >= 1");
        }

        float[] r = new float[arrayLength];

        for(int i = 0; i < r.length; i++){
            r[i] = Utility.getRandomUniform(lowerBound, upperBound);
        }

        return r;
    }


    /**
     * Initializes the parameters to a uniform distribution of [min, max)
     * @param parameters The parameters to initialize
     * @param min The minimum value an entry can be.
     * @param max The maximum value an entry can be.
     */
    public static void initializeUniform(ArrayList<float[][]> parameters, float min, float max){
        Utility.checkNotNull(parameters);

        if(max < min){
            throw new AssertionError("max parameter is smaller than the min parameter.");
        }

        for(int i = 0; i < parameters.size(); i++){
            initializeUniform(parameters.get(i), min, max);
        }

    }

    /**
     * Initializes the entries in the matrix to a unifrom distribution of [min, max).
     * @param matrix The matrix to initialize.
     * @param min The minimum value an entry can be.
     * @param max The maximum value an entry can be.
     */
    public static void initializeUniform(float[][] matrix, float min, float max){
        Utility.checkNotNull((Object)matrix);

        if(max < min){
            throw new AssertionError("max parameter is smaller than the min parameter.");
        }

        for(int r = 0; r < matrix.length; r++){
            Utility.initializeUniform(matrix[r], min, max);
        }
    }


    /**
     * Initializes entries in the array to a uniform distribution of [min, max).
     * @param array The array to initialize.
     * @param min The minimum value an entry can be.
     * @param max The maximum value an entry can be.
     */
    public static void initializeUniform(float[] array, float min, float max){
        Utility.checkNotNull(array);

        if(max < min){
            throw new AssertionError("max parameter is smaller than the min parameter.");
        }

        for(int i = 0; i < array.length; i++){
            array[i] = Utility.getRandomUniform(min, max);
        }
    }

    /**
     * Initializes the parameters to a samples from a normal distribution
     * @param parameters The parameters to initialize.
     * @param mean The mean of the normal distribution.
     * @param variance The variance of the normal distribution.
     */
    public static void initializeNormal(ArrayList<float[][]> parameters, float mean, float variance){
        Utility.checkNotNull(parameters);

        if(variance < 0){
            throw new AssertionError("variance parameter is less than zero when it should be >= 0");
        }

        Random r = new Random();

        for(int i = 0; i < parameters.size(); i++){
            initializeNormal(parameters.get(i), mean, variance, r);
        }
    }

    /**
     * Initializes the matrix elements to be sample from a normal distribution.
     * @param matrix The matrix to initialize.
     * @param mean The mean of the normal distribution.
     * @param variance The variance of the normal distribution.
     * @param randObject The Random object to get the random samples from.
     */
    public static void initializeNormal(float[][] matrix, float mean, float variance, Random randObject){
        Utility.checkNotNull(matrix, randObject);

        if(variance < 0){
            throw new AssertionError("variance parameter is less than zero when it should be >= 0");
        }

        for(int r = 0; r < matrix.length; r++){
            for(int c = 0; c < matrix[r].length; c++){
                float randomFloat = (float)randObject.nextGaussian();

                matrix[r][c] = (randomFloat * variance) - mean;
            }
        }
    }


    /**
     * Reads a text file and places all of the contents into a string
     * @param filePath The filepath to the text file to read.
     * @return The contents of the file as a string. Returns null and prints error message if file is
     * unable to be read.
     */
    public static String getTextFileContents(String filePath){
        Utility.checkNotNull(filePath);

        try{
            BufferedReader br = new BufferedReader(new FileReader(filePath));

            StringBuilder sb = new StringBuilder();

            String line = br.readLine();
            
            while(line != null){
                sb.append(line);
                line = br.readLine();
                sb.append("\n");
            }

            br.close();

            return sb.toString();

        } catch(Exception e){
            System.err.println("Exception when trying to read a text file: " + e.getMessage());
        }
        return null;
    }


    /**
     * Reads a .csv file and places the contents into a 2d String array.
     * @param filePath A valid filepath to a .csv file.
     * @param lineDelimiter The character(s) that define where a line ends. Typically a newline (\n) character.
     * @param elementDelimiter The character(s) that define where a element within a line end. Typically a comma (,).
     * @return The 2nd String array, or null on error.
     */
    public String[][] readFromCSV(String filePath, String lineDelimiter, String elementDelimiter){
        Utility.checkNotNull(filePath, lineDelimiter, elementDelimiter);

        String fileContents = getTextFileContents(filePath);
        
        if(fileContents == null){
            return null;
        }

        String[] lines = fileContents.split(lineDelimiter);

        String[][] r = new String[lines.length][];

        for(int i = 0; i < r.length; i++){
            String[] lineSplit = lines[i].split(elementDelimiter);

            r[i] = lineSplit;
        }

        return r;
    }

    /**
     * Reads a CSV file with a newline character for line delimiter and a comma character for element delimiter
     * and places the contents into a 2d String array.
     * @param filepath The filepath of the csv file. Should be a valid path.
     * @return The 2d String array with the .csv contents, or null if error.
     */
    public String[][] readFromCSV(String filepath){
        Utility.checkNotNull(filepath);

        return readFromCSV(filepath, "\n", ",");
    }


    /**
     * Copies the contents from one array into the other
     * @param source The array to copy contents to.
     * @param destination The array to copy contents from.
     */
    public static void copyArrayContents(float[] source, float[] destination){
        Utility.checkArrayLengthsEqual(source, destination);

        for(int i = 0; i < source.length; i++){
            destination[i] = source[i];
        }
    }

    /**
     * Finds the argument (index) that contains the largest value in the array.
     * @param array The array to search for the largest value in.
     * @return The index of the element with the largest value in the array.
     */
    public static int argMax(float[] array){
        Utility.checkNotNull(array);
        int maxIndex = 0;

        for(int i = 1; i < array.length; i++){
            if(array[i] > array[maxIndex]){
                maxIndex = i;
            }
        }

        return maxIndex;
    }
    
    /**
     * Throws an AssertionError when the any of the objects are null, else does nothing.
     * Should be used to check one or several objects where a null is a fatal error to the program (cannot recover).
     * @param objects The objects to check if null.
     */
    public static void checkNotNull(Object... objects){
        if(objects == null || objects.length == 0){
            return;
        }

        for(int i = 0; i < objects.length; i++){
            if(objects[i] == null){
                throw new AssertionError("Object is null where it should not be at position " + i);
            }
        }
    }


    /**
     * Throws an assertion error if the 2d array is ragged (some arrays are of different sizes), else does nothing.
     * Should be used if a ragged 2d array would cause a fatal error (can't recover program)
     * @param matrix The 2d array to check if ragged. Presumed to not be null.
     */
    public static void checkMatrixRectangle(float[][] matrix){
        int length = matrix[0].length;

        for(int i = 1; i < matrix.length; i++){
            if(matrix[i].length != length){
                throw new AssertionError("Matrix is not a rectangle where it should be.");
            }
        }
    }

    /**
     * Throws an assertion error if any of the matrix parameters are not a rectangle (all arrays must be the same size)
     * @param matrix The matrix parameters to check for
     */
    public static void checkMatrixRectangle(float[][]... matrix){
        if(matrix == null || matrix.length == 0){
            return;
        }

        for(int i = 0; i < matrix.length; i++){
            Utility.checkNotNull((Object)matrix[i]);
            Utility.checkMatrixRectangle(matrix[i]);
        }
    }


    /**
     * Checks the input matricies to see if the dimensions are equal (rows and columns).
     * Throws an exception if dimensions do not match Useful on elementwise operations.
     * @param matrix The matricies to check for same dimensions.
     */
    public static void checkMatrixDimensionsEqual(float[][]... matrix){
        if(matrix == null || matrix.length == 0){
            return;
        }

        Utility.checkNotNull((Object)matrix[0]);
        int rows = LinearAlgebra.getNumRows(matrix[0]);
        int cols = LinearAlgebra.getNumColumns(matrix[0]);

        for(int i = 1; i < matrix.length; i++){
            Utility.checkNotNull((Object)matrix[i]);

            int r = LinearAlgebra.getNumRows(matrix[i]);
            int c = LinearAlgebra.getNumColumns(matrix[i]);

            Utility.checkEqual(r, rows);
            Utility.checkEqual(c, cols);
        }
    }


    /**
     * Throws an assertion error if the input array is empty or has empty elements
     * where the length is equal to 0.
     * @param array The 2d array to check for being empty or having empty elements.
     */
    public static void checkArrayNotEmpty(float[][] array){
        if(array.length == 0){
            throw new AssertionError("2d array is of length 0.");
        }

        for(int i = 0; i < array.length; i++){
            Utility.checkArrayNotEmpty(array[i]);
        }
    }

    /**
     * Checks the input array to see if it empty (length of 0). If so, throws an assertion error.
     * @param array The array to check if empty.
     */
    public static void checkArrayNotEmpty(float[] array){
        if(array.length == 0){
            throw new AssertionError("2d array is of length 0.");
        }
    }

    /**
     * Checks to see if both list's 2d arrays are both not null and are of equal dimensions.
     * Throws an AssertionError if an issue is present.
     * @param listA The first list.
     * @param listB The second list.
     */
    public static void checkListDimensionsEqual(List<float[][]> listA, List<float[][]> listB){
        Utility.checkNotNull(listA, listB);

        if(listA.size() != listB.size()){
            throw new AssertionError("List sizes are not equal.");
        }

        for(int i = 0; i < listA.size(); i++){
            Utility.checkNotNull((Object)(listA.get(i)));
            Utility.checkNotNull((Object)(listB.get(i)));

            if(listA.get(i).length != listB.get(i).length){
                throw new AssertionError("Rows of matrix in list do not match.");
            }

            for(int j = 0; j < listA.get(i).length; j++){
                Utility.checkNotNull((Object)(listA.get(i)[j]));
                Utility.checkNotNull((Object)(listB.get(i)[j]));

                if(listA.get(i)[j].length != listB.get(i)[j].length){
                    throw new AssertionError("Columns of matrix in list do not match.");
                }
            }
        }
    }

    /**
     * Checks to see if the arrays are of the same length.
     * Will also check if they are null.
     * @param a The first array.
     * @param b The second array.
     */
    public static void checkArrayLengthsEqual(float[] a, float[] b){
        Utility.checkNotNull(a, b);

        Utility.checkEqual(a.length, b.length);
    }

    /**
     * Checks to see if the 2d arrays are of the same length (same number of rows).
     * Will also check if they are null.
     * @param a The first array.
     * @param b The second array.
     */
    public static void checkArrayLengthsEqual(float[][] a, float[][] b){
        Utility.checkNotNull((Object)a, (Object)b);

        Utility.checkEqual(a.length, b.length);
    }

    /**
     * Checks to see if both parameters are equal. If not, an assertion error is thrown.
     * @param a The first parameter to check.
     * @param b The second parameter to check.
     */
    public static void checkEqual(int a, int b){
        if(a != b){
            throw new AssertionError("Parameters are not equal when they should be.");
        }
    }

    /**
     * Checks the input floats to make sure it is real (not infinity or NaN).
     * If it's not real, throws an assertion error
     * @param x The input floats to check
     */
    public static void checkReal(float... x){
        if(x == null || x.length == 0){
            return;
        }

        for(int i = 0; i < x.length; i++){
            if(!Float.isFinite(x[i])){
                throw new AssertionError("Floating point number is not finite (either NaN, -inf, or inf)");
            }
        }
    }

    /**
     * Checks to see if input integers are greater than 0. Throws assertion error if <= 0
     * @param a The input integers
     */
    public static void checkGreaterThanZero(int... a){
        if(a == null || a.length == 0){
            return;
        }

        for(int i = 0; i < a.length; i++){
            if(a[i] <= 0){
                throw new AssertionError("Number should be > 0");
            }
        }
    }

    /**
     * Checks to see if the input arrays contain the same data. Can be used for tests
     * @param a The first array
     * @param b The second array
     * @return Ture if arrays are equal, else false.
     */
    public static boolean equal(float[] a, float[] b){
        Utility.checkNotNull((Object)a, (Object)b);

        if(a.length != b.length){
            return false;
        }

        for(int i = 0; i < a.length; i++){
            if(a[i] != b[i]){
                return false;
            }
        }

        return true;
    }


    /**
     * Checks to see if the input arrays contain the same data. Can be used for tests.
     * @param a The first array
     * @param b The second array
     * @return True if arrays are equal, else false.
     */
    public static boolean equal(float[][] a, float[][] b){
        Utility.checkNotNull((Object)a, (Object)b);

        if(a.length != b.length){
            return false;
        }

        for(int i = 0; i < a.length; i++){
            if(!Utility.equal(a[i], b[i])){
                return false;
            }
        }

        return true;
    }
}