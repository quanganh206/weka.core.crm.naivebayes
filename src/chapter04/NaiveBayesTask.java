package chapter04;

import weka.core.converters.CSVLoader;
import java.io.File;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.RemoveType;
import weka.filters.Filter;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import java.util.Random;
import weka.classifiers.bayes.NaiveBayes;

public class NaiveBayesTask {
	public static void main(String args[]) throws Exception {
		System.out.println("Start processing...");
		
		Classifier baselineNB = new NaiveBayes();
		double resNB[] = evaluate(baselineNB);
		System.out.println("Naive Bayes\n" + "\tchurn: " + resNB[0] + "\n" + "\tappetency: " + resNB[1] + "\n" + "\tup-sell: " + resNB[2] + "\n" + "\toverall: " + resNB[3] + "\n");

		System.out.println("End processing...");
	}

	public static double[] evaluate(Classifier model) throws Exception {
		double results[] = new double[4];

		String[] labelFiles = new String[] { "churn", "appetency", "upselling" };

		double overallScore = 0.0;

		for (int i = 0; i < labelFiles.length; i++) {
			// load data
			Instances train_data = loadData("data/orange_small_train.data",
					"data/orange_small_train_" + labelFiles[i] + ".labels.txt");

			// cross-validate the data
			Evaluation eval = new Evaluation(train_data);
			eval.crossValidateModel(model, train_data, 10, new Random(1));

			// save the results
			results[i] = eval.areaUnderROC(train_data.classAttribute().indexOfValue("1"));

			overallScore += results[i];
		}

		// get the average results over all three problems
		results[3] = overallScore / 3;

		return results;
	}

	public static Instances loadData(String pathData, String pathLabels) throws Exception {
		// load data
		CSVLoader loader = new CSVLoader();
		loader.setFieldSeparator("\t");
		loader.setNominalAttributes("191-last");
		loader.setSource(new File(pathData));
		Instances data = loader.getDataSet();

		// remove string attributes type
		RemoveType removeString = new RemoveType();
		removeString.setOptions(new String[] { "-T", "string" });
		removeString.setInputFormat(data);
		Instances filteredData = Filter.useFilter(data, removeString);

		// load labels
		loader = new CSVLoader();
		loader.setFieldSeparator("\t");
		loader.setNoHeaderRowPresent(true);
		loader.setNominalAttributes("first-last");
		loader.setSource(new File(pathLabels));
		Instances labels = loader.getDataSet();

		// append the labels as class value
		Instances labeledData = Instances.mergeInstances(filteredData, labels);

		labeledData.setClassIndex(labeledData.numAttributes() - 1);

		return labeledData;
	}
}
