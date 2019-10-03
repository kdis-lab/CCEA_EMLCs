package eaglet.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import eaglet.individualCreator.EagletIndividualCreator;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import net.sf.jclec.IIndividual;
import net.sf.jclec.IProvider;
import net.sf.jclec.ISpecies;
import net.sf.jclec.binarray.MultipBinArrayIndividual;
import net.sf.jclec.binarray.MultipBinArraySpecies;
import net.sf.jclec.fitness.SimpleValueFitness;
import net.sf.jclec.util.random.IRandGen;
import weka.core.Instance;
import weka.core.Instances;

public class Utils {
	
	/**
	 * Types of evaluation of the individuals.
	 * 	Train: Each individual is evaluated over their own train data
	 * 	Full: Individuals are eavaluated over full train data
	 * @author Jose
	 *
	 */
	public enum EvalType{
		train, full,
	};
	
	/**
	 * Techniques for sampling data
	 * 	pct67: 67% of instances (without replacement) are used for training
	 * 	pct75: 75% of instances (without replacement) are used for training
	 *  pct80: 80% of instances (without replacement) are used for training
	 */
	public enum SamplingTechnique{
		pct67, pct75, pct80,
	};
	
	/**
	 * Types of communication between subpopulations
	 *	no: No communication between subpopulations during the evolution; only at the end the ensemble is generated
	 *	exchange: Promising individuals are copied to other subpopulations, while useless individuals are removed
	 *	operators: Specific genetic operators between subpopulations are used to share information.
	 */
	public enum CommunicationType{
		no, exchange, operators,
	};
	
	
	/**
	 * Generates an array of shuffled values from 0 to n-1
	 * 
	 * @param n Length of the array
	 * @param seed Seed for random numbers
	 * 
	 * @return Array with shuffled values
	 */
	public static int [] shuffledArray(int n, long seed){
		int [] array = new int[n];
		
		for(int i=0; i<n; i++){
			array[i] = i;
		}
		
		Random rand = new Random(seed);
		int swap, r;
		
		for(int i=0; i<n; i++){
			r = rand.nextInt(n);
			swap = array[r];
			array[r] = array[i];
			array[i] = swap;
		}
		
		return array;
	}
	
	/**
	 * Calculates the appearances of each label of the dataset
	 * 
	 * @param mlData Multi-label dataset
	 * 
	 * @return Array with the number of appearances of each label
	 */
	public static int [] calculateAppearances(MultiLabelInstances mlData){
		int nLabels = mlData.getNumLabels();
		int [] appearances = new int[nLabels];
		
		int [] labelIndices = mlData.getLabelIndices();
		
		for(Instance instance : mlData.getDataSet()){
			for(int l=0; l<nLabels; l++){
				appearances[l] += instance.value(labelIndices[l]);
			}
		}
		
		return appearances;
	}
	
	/**
	 * Calculate the accumulated frequency given a int array
	 * 
	 * @param appearances Int array with the frequencies
	 * 
	 * @return Accumulated frequency
	 */
	public static double [] calculateFrequencies(int [] array){
		int total = 0;
		for(int i=0; i<array.length; i++){
			total += array[i];
		}
		
		double [] frequency = new double[array.length];
		for(int i=0; i<array.length; i++){
			frequency[i] = (double)array[i] / total;
		}
		
		return frequency;
	}
	
	/**
	 * Calculate the relative frequencies of appearance given a dataset
	 * 
	 * @param mlData Multi-label dataset
	 * 
	 * @return Relative frequencies of appearance of the labels
	 */
	public static double [] calculateFrequencies(MultiLabelInstances mlData){
		return calculateFrequencies(calculateAppearances(mlData));
	}

	/**
	 * Look if an array contains a value
	 * 
	 * @param array Int array
	 * @param n Value to find
	 * 
	 * @return True if the value exists and false otherwise
	 */
	public static boolean contains(int [] array, int n){
		for(int i=0; i<array.length; i++){
			if(array[i] == n){
				return true;
			}
		}
		
		return false;
	}
	
	/**
	 * Spread the votes evenly among the labels
	 * 
	 * @param numLabels Number of labels
	 * @param totalVotes Total number of votes to share
	 * @param seed Seed for random numbers
	 * 
	 * @return Array with evenly spreaded votes
	 */
	public static int [] spreadVotesEvenly(int numLabels, int totalVotes, int seed){	
		int [] expectedVotes = new int[numLabels];
		
		int share = totalVotes/numLabels;
		for(int i=0; i<numLabels; i++){
			expectedVotes[i] = share;
		}
		
		if((share*numLabels) < totalVotes){
			int toShare = totalVotes - (share*numLabels);
			
			int [] v = Utils.shuffledArray(numLabels, seed);
			
			int i=0;
			do{
				expectedVotes[v[i]]++;
				i++;
				toShare--;
			}while(toShare > 0);
		}
		
		return expectedVotes;
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

		for(int i=0; i<a1.length; i++){
			if(a1[i] != a2[i]){
				distance += 1;
			}
		}
		
		distance /= a1.length;
		
		return distance;
	}
	
	/**
	 * Weighted hamming distance between two arrays 
	 * 
	 * @param a1 Array 1
	 * @param a2 Array 2
	 * @param weights Array with weights to calculate the hamming distance. The greater is the weight, the greater is the distance for a given position
	 * 
	 * @return Weighted hamming distance
	 */
	public static double hammingDistance(byte [] a1, byte [] a2, double [] weights){
		double distance = 0;
		
		if(a1.length != a2.length){
			return -1;
		}
		
		for(int i=0; i<a1.length; i++){
			if(a1[i] != a2[i]){
				distance += weights[i];
			}
		}
		
		return distance;
	}
	
	/**
	 * Calculate hamming distance with weight between two MultipBinArrayIndividuals
	 * 
	 * @param ind1 First individual
	 * @param ind2 Secon individual
	 * @param weights Array with weights to calculate the hamming distance. The greater is the weight, the greater is the distance for a given position
	 * 
	 * @return Hamming distance with weights between the individuals
	 */
	public static double distance(MultipBinArrayIndividual ind1, MultipBinArrayIndividual ind2, double [] weights){		
		return hammingDistance(ind1.getGenotype(), ind2.getGenotype(), weights);
	}
	
	/**
	 * Looks if a given individual exists in the list of individuals
	 * 
	 * @param refInd Individual of reference to find
	 * @param list List of individuals
	 * 
	 * @return True if the individual exists and false otherwise
	 */
	public static boolean exists(IIndividual refInd, List<IIndividual> list){
		for(IIndividual ind : list){
			if(hammingDistance(((MultipBinArrayIndividual)refInd).getGenotype(), ((MultipBinArrayIndividual)ind).getGenotype()) == 0){
				return true;
			}
		}
		
		return false;
	}
	
	/**
	 * Looks if a given MultipBinArrayIndividual exists in the list of individuals
	 * 
	 * @param refInd Individual of reference to find
	 * @param list List of individuals
	 * 
	 * @return True if the individual exists and false otherwise
	 */
	public static boolean exists(MultipBinArrayIndividual refInd, List<MultipBinArrayIndividual> list){
		for(MultipBinArrayIndividual ind : list){
			if(hammingDistance(refInd.getGenotype(), ind.getGenotype()) == 0){
				return true;
			}
		}
		
		return false;
	}
	
	/**
	 * Remove duplicated individuals in a list
	 * 
	 * @param inds List of individuals
	 * 
	 * @return List of individuals with no duplicated
	 */
	public static List<IIndividual> removeDuplicated(List<IIndividual> inds){
		List<IIndividual> newList = new ArrayList<IIndividual>();
		
		for(IIndividual ind : inds){
			if(!Utils.exists(ind, newList)){
				newList.add(ind);
			}
		}
		
		return newList;
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
	 * Returns the index of the max value (or a random one if there is a tie)
	 * 
	 * @param array Array of int values
	 * @param seed Seed for random numbers
	 * 
	 * @return Index of the maximum value
	 */
	public static int getMaxIndex(int[] array, long seed){
		List<Integer> list = new ArrayList<>();
		
		int max = array[0];
		list.add(0);
		
		for(int i=1; i<array.length; i++){
			if(array[i] > max){
				max = array[i];
				list.clear();
				list.add(i);
			}
			else if(array[i] == max){
				list.add(i);
			}
		}
		
		if(list.size() > 1){
			Random rand = new Random(seed);
			int r = rand.nextInt(list.size());
			return list.get(r);
		}
		else{
			return(list.get(0));
		}
	}
	
	/**
	 * Returns the index of the max value (or a random one if there is a tie)
	 * 
	 * @param array Array of double values
	 * @param seed Seed for random numbers
	 * 
	 * @return Index of the maximum value (or random index between maximums)
	 */
	public static int getMaxIndex(double[] array, long seed){
		List<Integer> list = new ArrayList<>();
		
		double max = array[0];
		list.add(0);
		
		for(int i=1; i<array.length; i++){
			if(array[i] > max){
				max = array[i];
				list.clear();
				list.add(i);
			}
			else if(array[i] == max){
				list.add(i);
			}
		}
		
		if(list.size() > 1){
			Random rand = new Random(seed);
			int r = rand.nextInt(list.size());
			return list.get(r);
		}
		else{
			return(list.get(0));
		}
	}
	
	/**
	 * Returns the index of the min value that has not been selected (or a random one if there is a tie)
	 * 
	 * @param array Array of int values
	 * @param selected Array of selected values
	 * @param seed Seed for random numbers
	 * @return Index
	 */
	public static int getMinIndex(int[] array, byte[] selected, long seed){
		List<Integer> list = new ArrayList<>();
		
		int min = Integer.MAX_VALUE;
		
		for(int i=0; i<array.length; i++){
			if(selected[i] == 0){
				if(array[i] < min){
					min = array[i];
					list.clear();
					list.add(i);
				}
				else if(array[i] == min){
					list.add(i);
				}
			}			
		}
		
		if(list.size() > 1){
			Random rand = new Random(seed);
			int r = rand.nextInt(list.size());
			return list.get(r);
		}
		else{
			return(list.get(0));
		}
	}
	
	/**
	 * Selects a random individual from any population except from popToAvoid
	 * @param list
	 * @param popToAvoid
	 * @return
	 */
	public static IIndividual selectRandomIndividual(List<List<IIndividual>> list, int popToAvoid, IRandGen randgen) {
		int totalInds = 0;
		for(List<IIndividual> subpop : list) {
			if(((MultipBinArrayIndividual)subpop.get(0)).getSubpop() != popToAvoid) {
				totalInds += subpop.size();
			}
		}
		
		int index = randgen.choose(totalInds);
		for(List<IIndividual> subpop : list) {
			if(((MultipBinArrayIndividual)subpop.get(0)).getSubpop() != popToAvoid) {
				if(index >= subpop.size()) {
					index -= subpop.size();
				}
				else {
					return subpop.get(index);
				}
			}
		}
		
		return null;
	}
	
	/**
	 * Fill population (or subpopulation) with random individuals
	 * 
	 * @param pop List of individuals defininf a (sub)population
	 * @param toReach Number of individuals to reach in the population
	 */
	public static List<IIndividual> fillPopRandom(List<IIndividual> pop, int toReach, ISpecies species, IProvider provider){
		MultipBinArrayIndividual ind;
		int p = ((MultipBinArrayIndividual)pop.get(0)).getSubpop();
		
		while(pop.size() < toReach) {
			ind = ((MultipBinArraySpecies)species).createIndividual(((EagletIndividualCreator) provider).createRandomGenotype(), p);

			if(!containsComb(pop, ind)) {
				pop.add(ind);
			}
		}
		
		return pop;
	}
	
	/**
	 * Check if a list of individuals contains the combination of labels of a given individual
	 * 
	 * @param list List of individuals
	 * @param ind Single individual
	 * @return True if it contains and false otherwise
	 */
	public static boolean containsComb(List<IIndividual> list, IIndividual ind) {
		for(IIndividual oInd : list) {
			if(Arrays.toString(((MultipBinArrayIndividual)ind).getGenotype()).equals(Arrays.toString(((MultipBinArrayIndividual)oInd).getGenotype()))) {
				return true;
			}
		}
		return false;
	}
	
	/**
	 * Check if a list of MultipBinArrayIndividuals individuals contains a given individual
	 * 
	 * @param list List of individuals
	 * @param ind Given individual to check
	 * @return true if the individual exist in the list, false otherwise
	 */
	public static boolean contains(List<IIndividual> list, IIndividual ind) {
		for(IIndividual oInd : list) {
			if( ((MultipBinArrayIndividual)oInd).getSubpop() == ((MultipBinArrayIndividual)ind).getSubpop() ) {
				if(Arrays.toString(((MultipBinArrayIndividual)ind).getGenotype()).equals(Arrays.toString(((MultipBinArrayIndividual)oInd).getGenotype()))) {
					return true;
				}
			}
		}
		return false;
	}
	
	/**
	 * Partition data into train and validation sets
	 * 
	 * @param mlData Full data
	 * @param samplingTechnique Technique for selecting the data
	 * @return
	 */
	public static MultiLabelInstances sampleData(MultiLabelInstances mlData, SamplingTechnique samplingTechnique, IRandGen randgen){
		MultiLabelInstances newMLData = null;
		
		Instances data, newData;
		
		data = mlData.getDataSet();
		newData = new Instances(data);
		newData.removeAll(newData);
		int [] indexes = new int[data.numInstances()];
		for(int i=0; i<data.numInstances(); i++) {
			indexes[i] = i;
		}
		int r, aux;
		for(int i=0; i<data.numInstances(); i++) {
			r = randgen.choose(data.numInstances());
			aux = indexes[i];
			indexes[i] = indexes[r];
			indexes[r] = aux;
		}
		
		int limit; 
		
		switch (samplingTechnique) {
		case pct67:
			limit = (int)Math.round(data.numInstances()*.67);
			break;
		case pct75:
			limit = (int)Math.round(data.numInstances()*.67);
			break;
		case pct80:
			limit = (int)Math.round(data.numInstances()*.67);
			break;
			
		default:
			limit = -1;	
		}

		for(int i=0; i<limit; i++) {
			newData.add(data.get(indexes[i]));
		}
		
		try {
			newMLData = new MultiLabelInstances(newData, mlData.getLabelsMetaData());
		} catch (InvalidDataFormatException e1) {
			e1.printStackTrace();
		}
		
		return newMLData;
	}
	
	/**
	 * Get the individual with less predictive performance from those containing a given label
	 * 
	 * @param individuals List of individuals
	 * @param label Given label
	 * @return Worst individual with the label
	 */
	public static IIndividual getWorstIndividualByLabel(List<IIndividual> individuals, int label){
		List<IIndividual> candidates = new ArrayList<IIndividual>();
		
		for(int i=0; i<individuals.size(); i++){
			if(((MultipBinArrayIndividual)individuals.get(i)).getGenotype()[label] == 1){
				candidates.add(individuals.get(i));
			}
		}
		
		double minFitness = Double.MAX_VALUE;
		IIndividual worst = null;
		
		double currentFitness;
		for(int i=0; i<candidates.size(); i++){
			currentFitness = ((SimpleValueFitness)individuals.get(i).getFitness()).getValue();
			if(currentFitness < minFitness){
				if(! hasCriticalLabel(candidates.get(i), individuals)){
					minFitness = currentFitness;
					worst = candidates.get(i);
				}
			}
		}
		
		return worst;
	}
	
	/**
	 * Obtain all individuals containing a given label
	 * 
	 * @param list List of individuals
	 * @param label Label
	 * @return
	 */
	public static List<IIndividual> getIndividualsWithLabel(List<IIndividual> list, int label){
		List<IIndividual> candidates = new ArrayList<IIndividual>();
		for(int i=0; i<list.size(); i++){
			MultipBinArrayIndividual ind = (MultipBinArrayIndividual)list.get(i);
			byte [] genotype = ind.getGenotype();
			if(genotype[label] == 1){
				candidates.add(list.get(i));
			}
		}

		return candidates;
	}
	
	/**
	 * Indicates if an individual has a critical label.
	 * It is a critical label if it only appears once in the population
	 * 
	 * @param ind Individual to check
	 * @param list List of individuals
	 * @return True if any of the labels in ind is critical
	 */
	public static boolean hasCriticalLabel(IIndividual ind, List<IIndividual> list){
		int [] votesPerLabel = Utils.calculateVotesPerLabel(individualsToEnsembleMatrix(list));
		
		byte [] genotype = ((MultipBinArrayIndividual)ind).getGenotype();

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
	 * Calculate the distance from one individual to an ensemble
	 * 
	 * @param ind Individual
	 * @param ensemble Ensemble
	 * @param ensembleSize Size of the ensemble
	 * @param weights Weights for each label to calculate la distance
	 * @return Distance from individual to ensemble
	 */
	public static double distanceToEnsemble(IIndividual ind, List<IIndividual> ensemble, int ensembleSize, double [] weights){
		double distance = 0;
		
		for(int i=0; i<ensembleSize; i++){
			distance += Utils.distance((MultipBinArrayIndividual)ind, (MultipBinArrayIndividual)ensemble.get(i), weights);
		}
		
		distance /= ensembleSize;
		
		return distance;
	}
	
	/**
	 * Transform a list of individuals into an ensemble matrix
	 * 
	 * @param individuals List of individuals
	 * @return Byte matrix with the ensemble matrix
	 */
	public static byte [][] individualsToEnsembleMatrix(List<IIndividual> individuals){
		int numberLabels = ((MultipBinArrayIndividual)individuals.get(0)).getGenotype().length;
		byte [][] EnsembleMatrix = new byte[individuals.size()][numberLabels];
		
		for(int i=0; i<individuals.size(); i++){
			System.arraycopy(((MultipBinArrayIndividual)individuals.get(i)).getGenotype(), 0, EnsembleMatrix[i], 0, numberLabels);
		}
		
		return EnsembleMatrix;
	}
	
	/**
	 * Calculate the number of votes for each label in the list of individuals given.
	 * 
	 * @param inds List of individuals
	 * @param numLabels Number of labels
	 * @return Number of votes per label
	 */
	public static int[] calculateVotesPerLabel(List<IIndividual> inds, int numLabels)
	{	
		byte [][] EnsembleMatrix = individualsToEnsembleMatrix(inds);
		
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
}
