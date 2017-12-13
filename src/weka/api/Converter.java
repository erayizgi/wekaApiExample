package weka.api;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

import weka.core.Instances;
import weka.core.converters.CSVLoader;

public class Converter {
	public void Convert(String sourcepath,String destpath) throws Exception
    {
        // load CSV
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(sourcepath));
        Instances dataSet = loader.getDataSet();

        // save ARFF
        BufferedWriter writer = new BufferedWriter(new FileWriter(destpath));
        writer.write(dataSet.toString());
        writer.flush();
        writer.close(); 
    }

}
