import java.io.FileWriter;
import java.io.IOException;

/**
 * Class definition for CSVWriter, which allows users to easily write data to .csv files to be opened in other programs.
 */
public class CSVWriter {

    /**
     * The buffer used to store the string data before it is written to disk.
     * More optimized than simple concatenation.
     */
    private StringBuffer buffer;

    /**
     * The filepath where the csv file will be written to
     */
    private String filePath;

    /**
     * The number of columns in the csv file
     */
    private int numCols;

    /**
     * Constructs the CSVWriter object with user defined parameters.
     * @param filePath The filepath where the csv file will be written to.
     * @param columnHeaders The names for each of the columns which will show up as the first row in the csv file.
     */
    public CSVWriter(String filePath, String[] columnHeaders){
        //Check parameters
        Utility.checkNotNull(filePath, (Object)columnHeaders);

        for(int i = 0; i < columnHeaders.length; i++){
            Utility.checkNotNull(columnHeaders[i]);
        }

        //Populate the fields.
        this.filePath = filePath;

        this.numCols = columnHeaders.length;

        this.buffer = new StringBuffer();

        this.addRow(columnHeaders);
    }


    /**
     * @return The number of columns in the .csv file writer
     */
    public int getNumCols(){
        return this.numCols;
    }

    /**
     * @return The filepath that the .csv files are being written to
     */
    public String getFilePath(){
        return this.filePath;
    }

    /**
     * Adds a new row to the csv, but does not write the new row to disk.
     * @param newRow The new row. Should be the same size as the column header array
     * used to create the CSVWriter object.
     */
    public void addRow(String[] newRow){
        //Check parameter
        Utility.checkNotNull((Object)newRow);
        Utility.checkEqual(this.numCols, newRow.length);

        for(int i = 0; i < newRow.length - 1; i++){
            //Check to see if element in array is not null
            Utility.checkNotNull(newRow[i]);

            this.buffer.append(newRow[i] + ", ");
        }

        //Check last element for not being null
        Utility.checkNotNull(newRow[newRow.length - 1]);
        this.buffer.append(newRow[newRow.length - 1] + "\n");
    }

    /**
     * Writes the csv file stored in memory to the filepath on disk.
     * May display error message on failure.
     */
    public void writeToFile(){

        try{
            FileWriter writer = new FileWriter(this.filePath);
            writer.write(this.buffer.toString());
            writer.close();
        } catch(IOException exception){
            System.err.println("IO exception occured in writeToFile().");
            System.err.println(exception);
        }

    }

}