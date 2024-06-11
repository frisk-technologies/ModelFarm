import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class App {
    public static void main(String[] args) {
        CSVLoader loader = new CSVLoader();
        CSVLoader testLoader = new CSVLoader();
        Instances dataset;
        Instances testDataset;
        try {
            loader.setSource(new File("data/train.csv"));
            testLoader.setSource(new File("data/test.csv"));
            dataset = loader.getDataSet();
            testDataset = testLoader.getDataSet();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // Find attributes with missing values that are "NA" and replace them with "?" so that the ReplaceMissingValues filter can handle them
        HandleNAVals(dataset);

        // Same for the test dataset
        HandleNAVals(testDataset);

        // Now we need to make the same changes to the test dataset

        // Print the original attribute count and the attributes
        int originalAttributeCount = dataset.numAttributes();
        System.out.println("Original Attribute Count: " + originalAttributeCount);

        // Sigh, first off filling in the data. I don't really want to use medians this time.
        // Let's do it medians as well though why not.
        // TODO: Use more than just Median/Mode
        ReplaceMissingValues replaceMissingWithMMFilter = new ReplaceMissingValues();
        ReplaceMissingValues replaceMissingTestWithMMFilter = new ReplaceMissingValues();
        try {
            replaceMissingWithMMFilter.setInputFormat(dataset);
            replaceMissingTestWithMMFilter.setInputFormat(testDataset);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        Instances filledWithMM;
        Instances filledWithMMTest;
        try {
            filledWithMM = Filter.useFilter(dataset, replaceMissingWithMMFilter);
            filledWithMMTest = Filter.useFilter(testDataset, replaceMissingTestWithMMFilter);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        // Attribute count after MM filter
        int countAfterMM = filledWithMM.numAttributes();
        System.out.println("Num attributes after filling with MM: " + countAfterMM);

        // Split with 10-fold cross-validation
        filledWithMM.setClassIndex(filledWithMM.numAttributes() - 1);  // Set the index of the class attribute

        // Convert numerical to nominal
        NumericToNominal numericToNominalFilter = new NumericToNominal();
        NumericToNominal numericToNominalFilterTest = new NumericToNominal();
        numericToNominalFilter.setAttributeIndices("first-last");
        numericToNominalFilterTest.setAttributeIndices("first-last");
        try {
            numericToNominalFilter.setInputFormat(filledWithMM);
            numericToNominalFilterTest.setInputFormat(filledWithMMTest);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        Instances filteredDataset;
        Instances filteredTestDataset;
        try {
            filteredDataset = Filter.useFilter(filledWithMM, numericToNominalFilter);
            filteredTestDataset = Filter.useFilter(filledWithMMTest, numericToNominalFilterTest);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        // Count after converting numerical to nominal
        int countAfterNumericToNominal = filteredDataset.numAttributes();
        System.out.println("Num attributes after converting numerical to nominal: " + countAfterNumericToNominal);

        // Create a classifier (e.g., J48 decision tree)
        Classifier classifier = new J48();

        AdaBoostM1 ensemble = new AdaBoostM1();
        ensemble.setClassifier(classifier);

        // Set target attribute as the last attribute with the class index
        filteredDataset.setClassIndex(filteredDataset.numAttributes() - 1);

        // Remove any leftover string attributes from the dataset
        for (int i = 0; i < filteredDataset.numAttributes(); i++) {
            if (filteredDataset.attribute(i).isString()) {
                // Print what attribute is being removed
                System.out.println("Removing string attribute: " + filteredDataset.attribute(i).name());
                // Remove the attribute
                filteredDataset.deleteAttributeAt(i);
            }
        }
        // Do the same thing on the test dataset
        for (int i = 0; i < filteredTestDataset.numAttributes(); i++) {
            if (filteredTestDataset.attribute(i).isString()) {
                // Print what attribute is being removed
                System.out.println("Removing string attribute from test: " + filteredTestDataset.attribute(i).name());
                // Remove the attribute
                filteredTestDataset.deleteAttributeAt(i);
            }
        }

        // Print num after removing string attributes
        int countAfterRemovingString = filteredDataset.numAttributes();
        System.out.println("Num attributes after removing string attributes: " + countAfterRemovingString);

        // Perform 10-fold cross-validation training
        Evaluation eval;
        try {
            eval = new Evaluation(filteredDataset);
            eval.crossValidateModel(ensemble, filteredDataset, 10, new Random(1));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        // Print the accuracy and details of the best model
        double accuracy = eval.pctCorrect();
        System.out.println("Best Model Accuracy: " + accuracy);
        System.out.println("Stats: " + eval.toSummaryString());

        System.out.println();

        // Just to confirm, print the target attribute the model was trained to predict
        Attribute targetAttribute = filteredDataset.attribute(filteredDataset.classIndex());
        System.out.println("Target Attribute: " + targetAttribute.name());

        // Use model on test.csv and save result (id, saleprice) to submission.csv
        try {
            // Make sure m_ZeroR isn't null before classifying by correctly initializing the classifier
            ensemble.buildClassifier(filteredDataset);

            // the data must have exactly the same format (e.g., order of attributes) as the data used to train the classifier!
            // otherwise, the classifier will not work properly

            // Ensure that the test dataset has the same format as the training dataset
            if (filteredDataset.numAttributes() != filteredTestDataset.numAttributes()) {
                // Print info about the datasets to help debug

                // Print the number of attributes in the training dataset
                System.out.println("Num attributes in training dataset: " + filteredDataset.numAttributes());
                // Add each attribute to an array
                String[] trainingAttributes = new String[filteredDataset.numAttributes()];
                for (int i = 0; i < filteredDataset.numAttributes(); i++) {
                    trainingAttributes[i] = filteredDataset.attribute(i).name();
                }

                // Print the number of attributes in the test dataset
                System.out.println("Num attributes in test dataset: " + filteredTestDataset.numAttributes());
                // Add each attribute to an array
                String[] testAttributes = new String[filteredTestDataset.numAttributes()];
                for (int i = 0; i < filteredTestDataset.numAttributes(); i++) {
                    testAttributes[i] = filteredTestDataset.attribute(i).name();
                }
                // Print them side by side to compare
                for (int i = 0; i < trainingAttributes.length; i++) {
                    // Make sure it's in range or else it'll throw an exception
                    if (i < testAttributes.length) {
                        System.out.println(trainingAttributes[i] + " | " + testAttributes[i]);
                    } else {
                        System.out.println(trainingAttributes[i] + " | ");
                    }
                }

                throw new RuntimeException("Test dataset does not have the same format as the training dataset");
            }

            // Classify the test dataset
            double[] predictions = eval.evaluateModel(ensemble, filteredTestDataset);

            // Save the predictions to submission.csv
            CSVLoader submissionLoader = new CSVLoader();
            submissionLoader.setSource(new File("data/sample_submission.csv"));
            Instances submissionDataset = submissionLoader.getDataSet();

            for (int i = 0; i < submissionDataset.numInstances(); i++) {
                submissionDataset.instance(i).setValue(submissionDataset.attribute("SalePrice"), predictions[i]);
            }

            CSVSaver saver = new CSVSaver();
            saver.setInstances(submissionDataset);
            saver.setFile(new File("data/submission.csv"));
            saver.writeBatch();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static void HandleNAVals(Instances testDataset) {
        for (int i = 0; i < testDataset.numAttributes(); i++) {
            if (testDataset.attribute(i).isNumeric()) {
                for (int j = 0; j < testDataset.numInstances(); j++) {
                    if (testDataset.instance(j).isMissing(i)) {
                        testDataset.instance(j).setValue(i, 0);
                    }
                }
            }
        }
    }
}