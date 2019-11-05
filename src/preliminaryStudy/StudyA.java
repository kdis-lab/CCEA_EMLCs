package preliminaryStudy;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;

import coeaglet.algorithm.Ensemble;
import coeaglet.algorithm.EnsembleSelection;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.transformation.LabelPowerset2;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.measure.Measure;
import net.sf.jclec.IIndividual;
import net.sf.jclec.listind.MultipListIndividual;
import net.sf.jclec.util.random.IRandGen;
import net.sf.jclec.util.random.RanecuFactory;
import weka.classifiers.trees.J48;
import weka.core.Utils;


public class StudyA {
	
	public static void main(String [] args) {
		Hashtable<String, MultiLabelLearner> tableClassifiers = new Hashtable<String, MultiLabelLearner>();
		MultiLabelLearner learner = new LabelPowerset2(new J48());
		((LabelPowerset2)learner).setSeed(1);
		
		String trainName = null, testName = null, xmlName = null, outputName = null;
		int nSeeds=-1;
		int popSize=-1, numClassifiers=-1;
		PrintWriter pw = null;
		
		int aMin = 1;
		
		try {
			trainName = Utils.getOption('t', args);
			testName = Utils.getOption('T', args);
			xmlName = Utils.getOption('x', args);
			outputName = Utils.getOption('o', args);
			nSeeds = Integer.parseInt(Utils.getOption('s', args));
			
			pw = new PrintWriter(new FileWriter(outputName, true));
			String parameters = "n_s; beta; ";
			pw.println("data; i; " + StudyUtils.getCSVHeader(parameters));
			pw.close();
		
			int [] numPops = {3, 4, 5};
			double [] betas = {0.25, 0.5, 0.75};
			
			MultiLabelInstances fullTrainData = null;
			MultiLabelInstances testData = null;
			MultiLabelInstances[] subpopTrain = null;
			MultiLabelInstances[] validationData = null;
			
			int [][] appearances;

			double pctData = 0.75;
			int numberLabels = -1;
			int k = 3;
		
			fullTrainData = new MultiLabelInstances(trainName, xmlName);
			testData = new MultiLabelInstances(testName, xmlName);
			
			numberLabels = fullTrainData.getNumLabels();
			
			numClassifiers = (int)Math.round(3.33*numberLabels);
			popSize = numClassifiers*2;
			System.out.println("pSize: " + popSize);
			
			IRandGen randgen;
			RanecuFactory ran = new RanecuFactory();
			
			int seed;
			for(int i=0; i<nSeeds; i++) {
				seed = i*10;
				ran.setSeed(seed);
				randgen = ran.createRandGen();
				
				System.out.print("i " + i + "; \t" + "nSubpops: ");
				
				for(int p : numPops) {
					System.out.print(p + ", ");
					
					subpopTrain = new MultiLabelInstances[p];
					appearances = new int[p][];
					List<List<MultipListIndividual>> pop = new ArrayList<>(p);
					int subpopSize = (int)Math.ceil(popSize*1.0 / p);
					System.out.println("spSize: " + subpopSize);
					
					for(int j=0; j<p; j++) {
						//Select train subset for subpop
						subpopTrain[j] = coeaglet.utils.Utils.sampleData(fullTrainData, pctData, randgen);
						appearances[j] = coeaglet.utils.Utils.getAppearances(subpopTrain[j]);
						List<MultipListIndividual> inds = StudyUtils.generateIndividuals(subpopSize, j, appearances[j], aMin, k, randgen);
						pop.add(j, inds);
					}
					
					tableClassifiers.clear();
					validationData = new MultiLabelInstances[p];
					for(int j=0; j<p; j++) {
						validationData[j] = fullTrainData;
					}
					PopEvaluator popEval = new PopEvaluator();
					popEval.evaluatePop(pop, subpopTrain, validationData, learner, tableClassifiers);
					
					List<IIndividual> allInds = new ArrayList<IIndividual>();
					for(int j=0; j<p; j++)  {
						allInds.addAll(pop.get(j));
					}
					
					for(double b : betas) {
						EnsembleSelection eSel = new EnsembleSelection(allInds, subpopSize, numberLabels, b);
						eSel.setRandgen(randgen);
						eSel.selectEnsemble();
						
						learner = new LabelPowerset2(new J48());
						((LabelPowerset2)learner).setSeed(1);
						Ensemble ensemble = new Ensemble(eSel.getEnsemble(), learner);
						ensemble.setTableClassifiers(tableClassifiers);
						
						ensemble.build(fullTrainData);
						
						eSel = null;
						
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
				     	pw = new PrintWriter(new FileWriter(outputName, true));
						pw.println(trainName + "; " + i + "; ns_" + p + "; beta_" + b + "; " + results.toCSV());
						pw.close();
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
	
	}
	

}
