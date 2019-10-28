package preliminaryStudy;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.transformation.LabelPowerset2;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.measure.Measure;
import net.sf.jclec.binarray.*;
import weka.classifiers.trees.J48;
import weka.core.Utils;


public class StudyB {
	
	public static void main(String [] args) {
		
		/*
		Hashtable<String, MultiLabelLearner> tableClassifiers = new Hashtable<String, MultiLabelLearner>();
		MultiLabelLearner learner = new LabelPowerset2(new J48());
		((LabelPowerset2)learner).setSeed(1);
		
		String trainName = null, testName = null, xmlName = null, outputName = null, evalTypeStr = null;
		int nSeeds=-1, nPops=-1;
		double beta=-1.0;
		int popSize=-1;
		int [] numClassifiers;
		PrintWriter pw = null;
		
		try {
			trainName = Utils.getOption('t', args);
			testName = Utils.getOption('T', args);
			xmlName = Utils.getOption('x', args);
			outputName = Utils.getOption('o', args);
			nPops = Integer.parseInt(Utils.getOption('p', args));
			beta = Double.parseDouble(Utils.getOption('b', args));
			evalTypeStr = Utils.getOption('e', args);
			nSeeds = Integer.parseInt(Utils.getOption('s', args));
			
			pw = new PrintWriter(new FileWriter(outputName, true));
			String parameters = "popSize; evalType; beta; nClassifiers; pruned; ";
			pw.println("data; i; " + StudyUtils.getCSVHeader(parameters));
		
			//int [] numPops = {3, 4, 5};
			//StudyUtils.EvalType [] evalTypes = {StudyUtils.EvalType.full, StudyUtils.EvalType.train};
			//double [] betas = {0.1, 0.25, 0.5, 0.75, 0.9};
			
			StudyUtils.EvalType evalType = null;
			if(evalTypeStr.equalsIgnoreCase("train")) {
				evalType = EvalType.train;
			}
			else if(evalTypeStr.equalsIgnoreCase("full")){
				evalType = EvalType.full;
			}
			else {
				System.out.println("Incorrect evaluation type.");
				System.exit(1);
			}
			
			MultiLabelInstances fullTrainData = null;
			MultiLabelInstances testData = null;
			MultiLabelInstances[] subpopTrain = null;
			MultiLabelInstances[] validationData = null;

			double pctData = 0.75;
			int numberLabels = -1;
			int k = 3;
		
			fullTrainData = new MultiLabelInstances(trainName, xmlName);
			testData = new MultiLabelInstances(testName, xmlName);
			
			numberLabels = fullTrainData.getNumLabels();
			
			numClassifiers = new int[2];
			numClassifiers[0] = (int)Math.round(2*numberLabels);
			numClassifiers[1] = (int)Math.round(3.33*numberLabels);
			
			boolean [] prune = {false, true};
			
			
			int seed;
			for(int i=0; i<nSeeds; i++) {
				seed = i*10;
				System.out.print("i " + i + "; \t" + "nClassifiers: ");
				
				tableClassifiers.clear();
				subpopTrain = StudyUtils.sampleDataSubpops(fullTrainData, pctData, nPops);
				
				for(int n=0; n<numClassifiers.length; n++) {
					System.out.print(numClassifiers[n] + ", ");
					popSize = numClassifiers[n]*2;
					List<List<MultipBinArrayIndividual>> pop = StudyUtils.createPop(popSize, nPops, numberLabels, k, seed);
					
					PopEvaluator popEval = new PopEvaluator();;
					switch (evalType) {
					case train:
						validationData = null;
						popEval.evaluatePop(pop, subpopTrain, learner, tableClassifiers);
						break;

					case full:
						validationData = new MultiLabelInstances[nPops];
						for(int j=0; j<nPops; j++) {
							validationData[j] = fullTrainData;
						}
						popEval.evaluatePop(pop, subpopTrain, validationData, learner, tableClassifiers);
						break;
					}
					
					EnsembleGenerator eGen = new EnsembleGenerator();
					EnsembleMLC_Multip ensemble = null;
					
					for(boolean pr : prune) {
						ensemble = eGen.generateAndBuildEnsemble(pop, fullTrainData, numClassifiers[n], beta, learner, tableClassifiers, pr, seed);
						List<Measure> measures = new ArrayList<Measure>();  	       
			  	       	measures = StudyUtils.prepareMeasures(ensemble, testData);
				     	Evaluation results;     	
				     	Evaluator eval = new Evaluator();
				     	results = eval.evaluate(ensemble, testData, measures);
						pw.println(trainName + "; " + i + "; p_" + nPops + "; eval_" + evalType.name() + "; beta_" + beta + "; nClassifiers_" + numClassifiers[n] + "_" + ensemble.getEnsembleInds().size() + "; pr_" + pr + "; " + results.toCSV());
					}
					
				}
				
				System.out.println();
			}
			
			
			System.out.println("Finished!");
		}
		catch(Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
		finally{
    		if(pw != null)
    		{
    			pw.close();
    		}
		}
		
		*/
	}
	

}
