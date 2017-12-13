package weka.api;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class CrossValidate {
	public static void combineDataSets() throws Exception {
		for(int i = 1; i<=5; i++) {
			DataSource trainingSource = new DataSource("./dataset"+i+"/training.arff");
			DataSource source1 = new DataSource("./dataset"+i+"/testing.arff");
			Instances trainingDataset = trainingSource.getDataSet();
			Instances testDataset = source1.getDataSet();
			for(int x = 0; x< testDataset.numInstances(); x++) {
				Instance t = testDataset.instance(x);
				trainingDataset.add(t);
			}
			BufferedWriter writer = new BufferedWriter(new FileWriter("./dataset"+i+"/combined.arff"));
	        writer.write(trainingDataset.toString());
	        writer.flush();
	        writer.close(); 
		}
	}
	public static String doCrossValidation(Classifier cls,Instances ds) throws Exception {
		int folds = 10;
        Random rand = new Random(1);
		Evaluation eval = new Evaluation(ds);
		eval.crossValidateModel(cls, ds, folds, rand);
		String result =  eval.toSummaryString("=== " + folds + "-fold Cross-validation ===", false);
		double actuals = 0;
        double preds = 0;
        for(int n = 0; n < ds.numInstances(); n++) {
        	double actualValue = ds.instance(n).classValue();
      		Instance newInst = ds.instance((n!=ds.numInstances()-1)? n+1: n);
      		double predSMO = cls.classifyInstance(newInst);
      	
      		actuals += actualValue;
      		preds += predSMO;
        }
        double MR = (actuals-preds)/ds.numInstances();
        result +="MR  "+MR+ "\n";
		return result;
	}

}
