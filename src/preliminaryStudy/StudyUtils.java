package preliminaryStudy;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.measure.AveragePrecision;
import mulan.evaluation.measure.Coverage;
import mulan.evaluation.measure.ErrorSetSize;
import mulan.evaluation.measure.ExampleBasedAccuracy;
import mulan.evaluation.measure.ExampleBasedFMeasure;
import mulan.evaluation.measure.ExampleBasedPrecision;
import mulan.evaluation.measure.ExampleBasedRecall;
import mulan.evaluation.measure.ExampleBasedSpecificity;
import mulan.evaluation.measure.GeometricMeanAveragePrecision;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.HierarchicalLoss;
import mulan.evaluation.measure.IsError;
import mulan.evaluation.measure.MacroAUC;
import mulan.evaluation.measure.MacroFMeasure;
import mulan.evaluation.measure.MacroPrecision;
import mulan.evaluation.measure.MacroRecall;
import mulan.evaluation.measure.MacroSpecificity;
import mulan.evaluation.measure.MeanAveragePrecision;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.measure.MicroAUC;
import mulan.evaluation.measure.MicroFMeasure;
import mulan.evaluation.measure.MicroPrecision;
import mulan.evaluation.measure.MicroRecall;
import mulan.evaluation.measure.MicroSpecificity;
import mulan.evaluation.measure.OneError;
import mulan.evaluation.measure.RankingLoss;
import mulan.evaluation.measure.SubsetAccuracy;
import net.sf.jclec.binarray.*;
import weka.core.Instances;

public class StudyUtils {
	
	enum EvalType{
		train, full,
	};
	
	public static List<List<MultipBinArrayIndividual>> createPop(int popSize, int nPop, int numberLabels, int k, int seed){
		int subpopSize = (int)Math.ceil(popSize*1.0 / nPop);		
		List<List<MultipBinArrayIndividual>> inds = new ArrayList<>(nPop);
		
		for(int p=0; p<nPop; p++) {
			//System.out.println("\t\t\tCreate individuals of subpop " + p);
			//inds.add(p, new ArrayList<MultipBinArrayIndividual>());
			inds.add(p, provide(subpopSize, p, numberLabels, k, seed));
			checkAllLabels(inds.get(p), numberLabels, k, nPop, seed);
		}
		
		
		
		return inds;
	}
	
	/**
	 * Transform a list of individuals into an ensemble matrix
	 * 
	 * @param individuals List of individuals
	 * @return Byte matrix with the ensemble matrix
	 */
	public static byte [][] individualsToEnsembleMatrix(List<MultipBinArrayIndividual> individuals, int numberLabels){
		byte [][] EnsembleMatrix = new byte[individuals.size()][numberLabels];
		
		for(int i=0; i<individuals.size(); i++){
			System.arraycopy(((MultipBinArrayIndividual)individuals.get(i)).getGenotype(), 0, EnsembleMatrix[i], 0, numberLabels);
		}
		
		return EnsembleMatrix;
	}
	
	/**
	 * Calculate the number of votes per label in the ensemble
	 * 
	 * @return Array with the number of votes per label
	 */
	public static int[] calculateVotesPerLabel(byte [][] EnsembleMatrix)
	{	
		int numLabels = EnsembleMatrix[0].length;
		int [] votesPerLabel = new int[numLabels];
		
		for(int i=0; i<EnsembleMatrix.length; i++)
		{
			for(int j=0; j<EnsembleMatrix[0].length; j++)
			{
				votesPerLabel[j] += EnsembleMatrix[i][j];
			}
		}
		
		return votesPerLabel;
	}
	
	/**
	 * Get the individual with less predictive performance from those containing a given label
	 * 
	 * @param individuals List of individuals
	 * @param label Given label
	 * @return Worst individual with the label
	 */
	public static void changeMajorityLabelRandInd(List<MultipBinArrayIndividual> individuals, int maxLabel, int zeroLabel, int seed){
		List<Integer> cand = new ArrayList<Integer>();
		
		for(int i=0; i<individuals.size(); i++){
			if((individuals.get(i)).getGenotype()[maxLabel] == 1){
				cand.add(i);
			}
		}

		//MultipBinArrayIndividual worst = null;
		Random rand = new Random(seed);
		int r = rand.nextInt(cand.size());
		//System.out.println("r: " + r);
		
		//System.out.println(individuals.get(cand.get(r)).toString());
		individuals.get(cand.get(r)).getGenotype()[maxLabel] = 0;
		individuals.get(cand.get(r)).getGenotype()[zeroLabel] = 1;
		//System.out.println(individuals.get(cand.get(r)).toString());
		
	}
	
	/**
	 * Indicates if an individual has a critical label.
	 * It is a critical label if it only appears once in the population
	 * 
	 * @param ind Individual to check
	 * @param list List of individuals
	 * @return True if any of the labels in ind is critical
	 */
	public static boolean hasCriticalLabel(MultipBinArrayIndividual ind, List<MultipBinArrayIndividual> list){
		int numberLabels = ind.getGenotype().length;
		
		int [] votesPerLabel = eaglet.utils.Utils.calculateVotesPerLabel(individualsToEnsembleMatrix(list, numberLabels));
		
		byte [] genotype = ind.getGenotype();

		for(int i=0; i<votesPerLabel.length; i++){
			if(genotype[i] == 1){
				if(votesPerLabel[i] <= 1){
					return true;
				}
			}
		}
		
		return false;
	}
	
	/**
	 * Obtain all individuals containing a given label
	 * 
	 * @param indsCopy List of individuals
	 * @param label Label
	 * @return
	 */
	public static List<MultipBinArrayIndividual> getIndividualsWithLabel(List<MultipBinArrayIndividual> indsCopy, int label){
		List<MultipBinArrayIndividual> candidates = new ArrayList<MultipBinArrayIndividual>();
		//System.out.println(indsCopy.size());
		for(int i=0; i<indsCopy.size(); i++){
			MultipBinArrayIndividual ind = indsCopy.get(i);
			if(ind != null) {
				//System.out.println("\t" + i + ": " + ind.toString());
				byte [] genotype = ind.getGenotype();
				if(genotype[label] == 1){
					candidates.add(indsCopy.get(i));
				}
			}
			
		}

		return candidates;
	}
	
	public static void checkAllLabels(List<MultipBinArrayIndividual> inds, int numberLabels, int k, int nSubpops, int seed) {
		ArrayList<Integer> noVotesLabels = new ArrayList<Integer>();
		do{
			noVotesLabels.clear();
			int [] votesPerLabel = calculateVotesPerLabel(individualsToEnsembleMatrix(inds, numberLabels));

			for(int i=0; i<votesPerLabel.length; i++){
				if(votesPerLabel[i] == 0){
					noVotesLabels.add(i);
				}
			}

			if(noVotesLabels.size() > 0){
				//System.out.println("\n -> " + noVotesLabels.toString());
				
				Random rand = new Random(seed);
				
				int r = rand.nextInt(noVotesLabels.size());

				int currentLabel = noVotesLabels.get(r);

				//Remove the worst individual of the most voted label (if do not remove other label appearing only once
				changeMajorityLabelRandInd(inds, eaglet.utils.Utils.getMaxIndex(votesPerLabel, rand.nextInt(100)), currentLabel, rand.nextInt(100));
			}
		}while(noVotesLabels.size() > 0);
		
	}
	
	public static byte [] createIndWithLabel(int numLabels, int label, int k, int seed)
	{
		Random rand = new Random(seed);
		
		byte [] genotype = new byte[numLabels];

        int r, active = 0;
        
        genotype[label] = 1;
        active++;
        
		do{
            r = rand.nextInt(numLabels);
            if(genotype[r] != 1){
                genotype[r] = 1;
                active++;
            }
        }while(active < k);

		return genotype;
	}
	
	public static List<MultipBinArrayIndividual> provide(int numberOfIndividuals, int p, int numberLabels, int k, int seed)
	{
		// Result list
		List<MultipBinArrayIndividual> createdBuffer = new ArrayList<MultipBinArrayIndividual> (numberOfIndividuals);

		// Provide individuals
		MultipBinArrayIndividual ind;
		seed *= p;
		boolean exist;
		for (int i=0; i<numberOfIndividuals; i++) {
			//System.out.println("\t" + i);
			do{
				ind = new MultipBinArrayIndividual(createRandomGenotype(numberLabels, k, seed), p);
				//System.out.println("\t\t" + ind.toString());
				seed++;
				exist = exists(ind, createdBuffer);
			}while(exist);

			createdBuffer.add(ind);
		}
		
		
		// Returns result
		return createdBuffer;
	}
	
	/**
	 * Generates a random genotype
	 * 
	 * @return Random genotype
	 */
	public static byte [] createRandomGenotype(int numLabels, int k, int seed)
	{
		Random rand = new Random(seed);
		
		byte [] genotype = new byte[numLabels];

        int r, active = 0;
		do{
            r = rand.nextInt(numLabels);
            if(genotype[r] != 1){
                genotype[r] = 1;
                active++;
            }
        }while(active < k);

		return genotype;
	}
	
	

	
	public static MultiLabelInstances[] sampleDataSubpops(MultiLabelInstances mlData, double pct, int numPops) {
		MultiLabelInstances[] data = new MultiLabelInstances[numPops];
		
		for(int p=0; p<numPops; p++) {
			data[p] = sampleData(mlData, pct, p*10);
		}
		
		return data;
	}
	
	public static MultiLabelInstances sampleData(MultiLabelInstances mlData, double pct, int seed) {
		Random rand = new Random(seed);
		
		Instances data = mlData.getDataSet();
		Instances newData = new Instances(data);
		newData.removeAll(newData);
		
		int [] indexes = new int[data.numInstances()];
		for(int i=0; i<data.numInstances(); i++) {
			indexes[i] = i;
		}
		
		int r, aux;
		for(int i=0; i<data.numInstances(); i++) {
			r = rand.nextInt(data.numInstances());
			aux = indexes[i];
			indexes[i] = indexes[r];
			indexes[r] = aux;
		}
		
		double limit = (int)Math.round(data.numInstances()*pct);
		
		for(int i=0; i<limit; i++) {
			newData.add(data.get(indexes[i]));
		}
		
		MultiLabelInstances sampledData = null;
		try {
			sampledData = new MultiLabelInstances(newData, mlData.getLabelsMetaData());
		} catch (InvalidDataFormatException e1) {
			e1.printStackTrace();
		}
		
		return sampledData;
	}
	
	public static boolean exists(MultipBinArrayIndividual refInd, List<MultipBinArrayIndividual> list){
		for(MultipBinArrayIndividual ind : list){
			if(hammingDistance(refInd.getGenotype(), ind.getGenotype()) == 0){
				return true;
			}
		}
		
		return false;
	}
	
	/**
	 * Hamming distance between two byte arrays
	 * 
	 * @param a1 array 1
	 * @param a2 array 2
	 * 
	 * @return Hamming distance
	 */
	public static double hammingDistance(byte [] a1, byte [] a2){
		double distance = 0;
		
		if(a1.length != a2.length){
			return -1;
		}
		
		double [] weights = new double[a1.length];
		for(int i=0; i<weights.length; i++){
			weights[i] = (double)1 / weights.length;
		}
		
		for(int i=0; i<a1.length; i++){
			if(a1[i] != a2[i]){
				distance += 1;
			}
		}
		
		distance /= a1.length;
		
		return distance;
	}
	
	/**
     * Prepare the measures to evaluate
     * 
     * @param learner Multi-label learner
     * @param mlTestData Multi-label data to evaluate
     * 
     * @return List with the measures
     */
    public static List<Measure> prepareMeasures(MultiLabelLearner learner,
            MultiLabelInstances mlTestData) {
        List<Measure> measures = new ArrayList<Measure>();

        MultiLabelOutput prediction;
        try {
            prediction = learner.makePrediction(mlTestData.getDataSet().instance(0));
            int numOfLabels = mlTestData.getNumLabels();
            
            // add bipartition-based measures if applicable
            if (prediction.hasBipartition()) {
                // add example-based measures
                measures.add(new HammingLoss());
                measures.add(new SubsetAccuracy());
                measures.add(new ExampleBasedPrecision());
                measures.add(new ExampleBasedRecall());
                measures.add(new ExampleBasedFMeasure());
                measures.add(new ExampleBasedAccuracy());
                measures.add(new ExampleBasedSpecificity());
                // add label-based measures
                measures.add(new MicroPrecision(numOfLabels));
                measures.add(new MicroRecall(numOfLabels));
                measures.add(new MicroFMeasure(numOfLabels));
                measures.add(new MicroSpecificity(numOfLabels));
                measures.add(new MacroPrecision(numOfLabels));
                measures.add(new MacroRecall(numOfLabels));
                measures.add(new MacroFMeasure(numOfLabels));
                measures.add(new MacroSpecificity(numOfLabels));
            }
            // add ranking-based measures if applicable
            if (prediction.hasRanking()) {
                // add ranking based measures
                measures.add(new AveragePrecision());
                measures.add(new Coverage());
                measures.add(new OneError());
                measures.add(new IsError());
                measures.add(new ErrorSetSize());
                measures.add(new RankingLoss());
            }
            // add confidence measures if applicable
            if (prediction.hasConfidences()) {
                measures.add(new MeanAveragePrecision(numOfLabels));
                measures.add(new GeometricMeanAveragePrecision(numOfLabels));
               // measures.add(new MeanAverageInterpolatedPrecision(numOfLabels, 10));
               // measures.add(new GeometricMeanAverageInterpolatedPrecision(numOfLabels, 10));
                measures.add(new MicroAUC(numOfLabels));
                measures.add(new MacroAUC(numOfLabels));
               // measures.add(new LogLoss());
            }
            // add hierarchical measures if applicable
            if (mlTestData.getLabelsMetaData().isHierarchy()) {
                measures.add(new HierarchicalLoss(mlTestData));
            }
        } catch (Exception ex) {
            Logger.getLogger(Evaluator.class.getName()).log(Level.SEVERE, null, ex);
        }

        return measures;
    }
    
    /**
     * Prepare the measures to evaluate
     * 
     * @param learner Multi-label learner
     * @param mlTestData Multi-label data to evaluate
     * 
     * @return List with the measures
     */
    public static String getCSVHeader(String parameters) {
    	String header = new String();
    	
    	header += parameters;
    	
    	header += new HammingLoss().getName() + "; ";
    	header += new SubsetAccuracy().getName() + "; ";
    	header += new ExampleBasedPrecision().getName() + "; ";
    	header += new ExampleBasedRecall().getName() + "; ";
    	header += new ExampleBasedFMeasure().getName() + "; ";
    	header += new ExampleBasedAccuracy().getName() + "; ";
    	header += new ExampleBasedSpecificity().getName() + "; ";
    	
    	header += new MicroPrecision(2).getName() + "; ";
    	header += new MicroRecall(2).getName() + "; ";
    	header += new MicroFMeasure(2).getName() + "; ";
    	header += new MicroSpecificity(2).getName() + "; ";
    	header += new MacroPrecision(2).getName() + "; ";
    	header += new MacroRecall(2).getName() + "; ";
    	header += new MacroFMeasure(2).getName() + "; ";
    	header += new MacroSpecificity(2).getName() + "; ";
    	
    	header += new AveragePrecision().getName() + "; ";
    	header += new Coverage().getName() + "; ";
    	header += new OneError().getName() + "; ";
    	header += new IsError().getName() + "; ";
    	header += new ErrorSetSize().getName() + "; ";
    	header += new RankingLoss().getName() + "; ";
    	
    	header += new MeanAveragePrecision(2).getName() + "; ";
    	header += new GeometricMeanAveragePrecision(2).getName() + "; ";
    	header += new MicroAUC(2).getName() + "; ";
    	header += new MacroAUC(2).getName() + "; ";
    	
    	//header += "HierarchicalLoss" + "; ";

        return header;
    }

}
