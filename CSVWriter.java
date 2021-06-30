import java.io.FileWriter;
import java.io.IOException;

/**
 * Class definition for CSVWriter, which allows users to easily write data to .csv files to be opened in other programs.
 */
public class CSVWriter {

    /**
     * 
     */
    private String dataString;

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
        this.filePath = filePath;

        this.numCols = columnHeaders.length;

        this.dataString = "";

        addRow(columnHeaders);
    }

    /**
     * Adds a new row to the csv, but does not write the new row to disk.
     * @param newRow The new row. Should be the same size as the column header array
     * used to create the CSVWriter object.
     */
    public void addRow(String[] newRow){
        for(int i = 0; i < newRow.length - 1; i++){
            dataString += newRow[i] + ", ";
        }

        dataString += newRow[newRow.length - 1] + "\n";
    }

    /**
     * Writes the csv file stored in memory to the filepath on disk.
     * May display error message on failure.
     */
    public void writeToFile(){

        try{
            FileWriter writer = new FileWriter(this.filePath);
            writer.write(this.dataString);
            writer.close();
        } catch(IOException exception){
            System.err.println("IO exception occured in writeToFile().");
            System.err.println(exception);
        }

    }


}