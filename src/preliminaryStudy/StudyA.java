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

import preliminaryStudy.algorithm.EnsembleMLC_Multip;

public class StudyA {
	
	public static void main(String [] args) {
		/*
		Hashtable<String, MultiLabelLearner> tableClassifiers = new Hashtable<String, MultiLabelLearner>();
		MultiLabelLearner learner = new LabelPowerset2(new J48());
		((LabelPowerset2)learner).setSeed(1);
		
		String trainName = null, testName = null, xmlName = null, outputName = null;
		int nSeeds=-1;
		int popSize=-1, numClassifiers=-1;
		PrintWriter pw = null;
		
		try {
			trainName = Utils.getOption('t', args);
			testName = Utils.getOption('T', args);
			xmlName = Utils.getOption('x', args);
			outputName = Utils.getOption('o', args);
			nSeeds = Integer.parseInt(Utils.getOption('s', args));
			
			pw = new PrintWriter(new FileWriter(outputName, true));
			String parameters = "popSize; evalType; beta; ";
			pw.println("data; i; " + StudyUtils.getCSVHeader(parameters));
		
			int [] numPops = {3, 4, 5};
			StudyUtils.EvalType [] evalTypes = {StudyUtils.EvalType.full, StudyUtils.EvalType.train};
			double [] betas = {0.1, 0.25, 0.5, 0.75, 0.9};
			
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
			
			numClassifiers = (int)Math.round(3.33*numberLabels);
			popSize = numClassifiers*2;
			
			int seed;
			for(int i=0; i<nSeeds; i++) {
				seed = i*10;
				System.out.print("i " + i + "; \t" + "nSubpops: ");
				
				for(int p : numPops) {
					System.out.print(p + ", ");
					//pw.println("\nNUMBER OF POPULATIONS: " + p);
					// Generate population for each subpop
					List<List<MultipBinArrayIndividual>> pop = StudyUtils.createPop(popSize, p, numberLabels, k, seed);
					
					//Select train subset for subpop
					subpopTrain = StudyUtils.sampleDataSubpops(fullTrainData, pctData, p);
					
					for(StudyUtils.EvalType e : evalTypes) {
						tableClassifiers.clear();
						
						// Set dataset to evaluate population
						PopEvaluator popEval = new PopEvaluator();;
						switch (e) {
						case train:
							//pw.println("\tEVAL TYPE: train");
							validationData = null;
							popEval.evaluatePop(pop, subpopTrain, learner, tableClassifiers);
							break;

						case full:
							//popEval = new PopEvaluator();
							//pw.println("\tEVAL TYPE: full");
							validationData = new MultiLabelInstances[p];
							for(int j=0; j<p; j++) {
								validationData[j] = fullTrainData;
							}
							popEval.evaluatePop(pop, subpopTrain, validationData, learner, tableClassifiers);
							break;
						}
						
						for(double b : betas) {
							//pw.println("\t\tBETA: " + b);
							// Generate ensemble
							//System.out.println("\t\t\tGenerating ensemble.");
							
							EnsembleGenerator eGen = new EnsembleGenerator();
							EnsembleMLC_Multip ensemble = eGen.generateAndBuildEnsemble(pop, fullTrainData, numClassifiers, b, learner, tableClassifiers, false, seed);
							
							//ensemble.printEnsemble();
							
							// Evaluate ensemble over full training 
							//System.out.println("\t\t\tEvaluating ensemble.");
							List<Measure> measures = new ArrayList<Measure>();  	       
				  	       	measures = StudyUtils.prepareMeasures(ensemble, testData);
					     	Evaluation results;     	
					     	Evaluator eval = new Evaluator();
					     	
					     	//ensemble.printEnsemble();
					     	//System.out.println("size: " + ensemble.getEnsembleInds().size());
					     	results = eval.evaluate(ensemble, testData, measures);
					     	
							
							//pw.println("\t\t\tFitness: " + fitness);
							pw.println(trainName + "; " + i + "; p_" + p + "; eval_" + e.name() + "; beta_" + b + "; " + results.toCSV());
						}
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
