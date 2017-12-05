package chapter04;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import java.io.File;
import java.time.Duration;
import java.time.Instant;

import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RemoveType;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;

import java.util.Random;

import weka.classifiers.EnsembleLibrary;
import weka.classifiers.meta.EnsembleSelection;
import weka.filters.unsupervised.attribute.RemoveUseless;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Discretize;

import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.Ranker;

public class EnsembleTask {
	public static void main(String args[]) throws Exception {
		System.out.println("Start processing...");
		Instant start = Instant.now();
				
		EnsembleLibrary ensembleLib = new EnsembleLibrary();

		// Decision trees
		ensembleLib.addModel("weka.classifiers.trees.J48 -S -C 0.25 -B -M 2");
		ensembleLib.addModel("weka.classifiers.trees.J48 -S -C 0.25 -B -M 2 -A");

		// naive Bayes
		ensembleLib.addModel("weka.classifiers.bayes.NaiveBayes");

		// k-nn
		ensembleLib.addModel("weka.classifiers.lazy.IBk");

		// AdaBoost
		ensembleLib.addModel("weka.classifiers.meta.AdaBoostM1");

		// LogitBoost
		ensembleLib.addModel("weka.classifiers.meta.LogitBoost");

		System.out.println(ensembleLib.getModels());
		EnsembleLibrary.saveLibrary(new File("data/ensembleLib.model.xml"), ensembleLib, null);

		EnsembleSelection ensambleSel = new EnsembleSelection();
		ensambleSel.setOptions(new String[] { "-L", "data/ensembleLib.model.xml", // </path/to/modelLibrary> - Specifies
																					// the Model Library File,
																					// continuing the list of all
																					// models.
				"-W", "data/esTmp", // </path/to/working/directory> - Specifies the Working Directory, where all
									// models will be stored.
				"-B", "10", // <numModelBags> - Set the number of bags, i.e., number of iterations to run
							// the ensemble selection algorithm.
				"-E", "1.0", // <modelRatio> - Set the ratio of library models that will be randomly chosen
								// to populate each bag of models.
				"-V", "0.25", // <validationRatio> - Set the ratio of the training data set that will be
								// reserved for validation.
				"-H", "100", // <hillClimbIterations> - Set the number of hillclimbing iterations to be
								// performed on each model bag.
				"-I", "1.0", // <sortInitialization> - Set the the ratio of the ensemble library that the
								// sort initialization algorithm will be able to choose from while initializing
								// the ensemble for each model bag
				"-X", "2", // <numFolds> - Sets the number of cross-validation folds.
				"-P", "roc", // <hillclimbMettric> - Specify the metric that will be used for model selection
								// during the hillclimbing algorithm.
				"-A", "forward", // <algorithm> - Specifies the algorithm to be used for ensemble selection.
				"-R", "true", // - Flag whether or not models can be selected more than once for an ensemble.
				"-G", "true", // - Whether sort initialization greedily stops adding models when performance
								// degrades.
				"-O", "true", // - Flag for verbose output. Prints out performance of all selected models.
				"-S", "1", // <num> - Random number seed.
				"-D", "true" // - If set, classifier is run in debug mode and may output additional info to
								// the console
		});

		double resES[] = evaluate(ensambleSel);
		System.out.println("Ensemble\n" + "\tchurn:     " + resES[0] + "\n" + "\tappetency: " + resES[1] + "\n"
				+ "\tup-sell:   " + resES[2] + "\n" + "\toverall:   " + resES[3] + "\n");

		Instant end = Instant.now();
		System.out.println("Processing Time: " + Duration.between(start, end));
		System.out.println("End processing...");
	}

	public static Instances preProcessData(Instances data) throws Exception {
		// remove useless attributes
		RemoveUseless removeUseless = new RemoveUseless();
		removeUseless.setOptions(new String[] { "-M", "99" });
		removeUseless.setInputFormat(data);
		data = Filter.useFilter(data, removeUseless);

		// fix missing values
		ReplaceMissingValues fixMissing = new ReplaceMissingValues();
		fixMissing.setInputFormat(data);
		data = Filter.useFilter(data, fixMissing);

		Discretize discretizeNumeric = new Discretize();
		discretizeNumeric.setOptions(new String[] { "-B", "4", // no of bins
				"-R", "first-last" });
		discretizeNumeric.setInputFormat(data);
		data = Filter.useFilter(data, discretizeNumeric);

		/**
		 * Select only informative attributes
		 */
		InfoGainAttributeEval eval = new InfoGainAttributeEval();
		Ranker search = new Ranker();
		search.setOptions(new String[] { "-T", "0.001" });

		AttributeSelection attSelect = new AttributeSelection();
		attSelect.setEvaluator(eval);
		attSelect.setSearch(search);

		// apply attribute selection
		attSelect.SelectAttributes(data);

		// remove the attributes not selected in the last run
		data = attSelect.reduceDimensionality(data);

		return data;
	}

	public static double[] evaluate(Classifier model) throws Exception {
		double[] results = new double[4];

		String[] labelFiles = new String[] { "churn", "appetency", "upselling" };

		double overallScore = 0.0;
		for (int i = 0; i < labelFiles.length; i++) {
			Instances train_data = loadData("data/orange_small_train.data",
					"data/orange_small_train_" + labelFiles[i] + ".labels.txt");

			// cross-validate the data
			Evaluation eval = new Evaluation(train_data);
			eval.crossValidateModel(model, train_data, 5, new Random(1));

			// save data
			results[i] = eval.areaUnderROC(train_data.classAttribute().indexOfValue("1"));
			overallScore += results[i];
		}

		// calculate average results over all three problem
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

		// remove String attribute type
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

		Instances labeledData = Instances.mergeInstances(filteredData, labels);
		labeledData.setClassIndex(labeledData.numAttributes() - 1);

		return labeledData;
	}
}
