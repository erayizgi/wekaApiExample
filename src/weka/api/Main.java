package weka.api;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.M5P;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

public class Main {

	public static void main(String[] args) {
		try {
			//CrossValidate.combineDataSets();
			
			System.out.println("_______ CASE STUDY 1 (Cross Validation) _______");
			for(int i = 1; i <=5; i++) {
			
				DataSource combinedDataset = new DataSource("./dataset"+i+"/combined.arff");
		        Instances combined = combinedDataset.getDataSet();
		        combined.setClassIndex(combined.numAttributes()-1);
		        Discretize dis = new Discretize();
		        String[] disOpt = new String[2];
		        disOpt[0] = "-R";
		        disOpt[1] = "2-3";
		        dis.setInputFormat(combined);
		        dis.setOptions(disOpt);
		        Instances filteredCombined = Filter.useFilter(combined, dis);
		        
		        LinearRegression lr = new LinearRegression();
		        lr.buildClassifier(filteredCombined);
		        
		        IBk knn = new IBk(1);
		        knn.buildClassifier(filteredCombined);
		        
		        SMOreg svm = new SMOreg();
		        svm.buildClassifier(filteredCombined);
		        
		        MultilayerPerceptron mlp = new MultilayerPerceptron();
				mlp.setLearningRate(0.3);
				mlp.setMomentum(0.2);
				mlp.setTrainingTime(500);
				mlp.setHiddenLayers("a");
				mlp.buildClassifier(filteredCombined);
				
				M5P tree = new M5P();
				tree.buildClassifier(filteredCombined);
		        
		        System.out.println("========= LINEAR REGRESSION CV DATASET "+i +" =========");
		        System.out.println(CrossValidate.doCrossValidation(lr, filteredCombined));
		        
		        System.out.println("========= KNN CV DATASET "+i +"  =========");
		        System.out.println(CrossValidate.doCrossValidation(knn, filteredCombined));
		        
		        System.out.println("========= SVM CV DATASET "+i +"  =========");
		        System.out.println(CrossValidate.doCrossValidation(svm, filteredCombined));
		        
		        System.out.println("========= MLP CV DATASET "+i +"  =========");
		        System.out.println(CrossValidate.doCrossValidation(mlp, filteredCombined));

		        System.out.println("========= M5P CV DATASET "+i +"  =========");
		        System.out.println(CrossValidate.doCrossValidation(tree, filteredCombined));
			}
			System.out.println("_______ CASE STUDY 2 (Normal Train and Test) _______");
			for(int i=1; i<= 5; i++) {
				DataSource source = new DataSource("./dataset"+i+"/training.arff");
		        Instances training = source.getDataSet();
		        DataSource source1 = new DataSource("./dataset"+i+"/testing.arff");
		        Instances testing = source1.getDataSet();
		        
		        training.setClassIndex(training.numAttributes()-1);
		        testing.setClassIndex(testing.numAttributes()-1);
		        
		        Discretize dis = new Discretize();
		        String[] disOpt = new String[2];
		        disOpt[0] = "-R";
		        disOpt[1] = "2-3";
		        dis.setInputFormat(training);
		        dis.setOptions(disOpt);
		        Instances filteredTraining = Filter.useFilter(training, dis);
		        
		        dis.setInputFormat(testing);
		        Instances filteredTesting = Filter.useFilter(testing, dis);
		        
		        LinearRegression lr = new LinearRegression();
		        lr.buildClassifier(filteredTraining);
		        
		        SMOreg svm = new SMOreg();
		        svm.buildClassifier(filteredTraining);
		        
		        M5P tree = new M5P();
		        tree.buildClassifier(filteredTraining);
		        
		        IBk knn = new IBk(1);
		        knn.buildClassifier(filteredTraining);
		        
		        MultilayerPerceptron mlp = new MultilayerPerceptron();
				mlp.setLearningRate(0.3);
				mlp.setMomentum(0.2);
				mlp.setTrainingTime(500);
				mlp.setHiddenLayers("2");
				mlp.buildClassifier(filteredTraining);
		        
		        System.out.println(evaluateModel(lr,filteredTraining,filteredTesting,"Linear Regression",i));
		        System.out.println(evaluateModel(svm,filteredTraining,filteredTesting,"SVM",i));
		        System.out.println(evaluateModel(tree,filteredTraining,filteredTesting,"M5P",i));
		        System.out.println(evaluateModel(knn,filteredTraining,filteredTesting,"KNN",i));
		        System.out.println(evaluateModel(mlp,filteredTraining,filteredTesting,"MLP",i));
			}
			
		}catch(Exception e) {
			
		}
		

	}
	public static String evaluateModel(Classifier cls,Instances trainingSet, Instances testingSet,String modelname,int datasetNumber) throws Exception {
		Evaluation eval = new Evaluation(trainingSet);
        eval.evaluateModel(cls, testingSet);
        String result =  eval.toSummaryString("=== "+modelname+" for Dataset "+datasetNumber+" ===", false);
        double actuals = 0;
        double preds = 0;
        for(int n = 0; n < testingSet.numInstances(); n++) {
        	double actualValue = testingSet.instance(n).classValue();
      		Instance newInst = testingSet.instance((n!=testingSet.numInstances()-1)? n+1: n);
      		double predSMO = cls.classifyInstance(newInst);
      	
      		actuals += actualValue;
      		preds += predSMO;
        }
        double MR = (actuals-preds)/testingSet.numInstances();
        result += "MR "+MR;
        return result;
	}

}
