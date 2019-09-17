package eaglet.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import mulan.data.MultiLabelInstances;
import net.sf.jclec.IIndividual;
import net.sf.jclec.binarray.MultipBinArrayIndividual;
import weka.core.Instance;

public class Utils {
	
	/**
	 * Selects an index based on weights -> Indices with more weight are more likely to be selected
	 *
	 * @param weights Double array of weights
	 * @return Selected index
	 */
	public static int selectBasedOnWeights(double [] weights, long seed){
		double [] acc = new double[weights.length];
		
		Random rand = new Random(seed);
		
		acc[0] = weights[0];
		for(int i=1; i<weights.length; i++){
			acc[i] = acc[i-1] + weights[i];
		}
		
		double r = rand.nextDouble() * acc[acc.length-1];
		
		for(int i=0; i<acc.length; i++){
			if(r <= acc[i]){
				return i;
			}
		}
		
		return -1;
	}

	/**
	 * Generates an array with the accumulated values of the original array
	 * 
	 * @param array Array of double values
	 * 
	 * @return Array with accumulated values
	 */
	public static double [] getAccumulatedArray (double [] array){
		double [] acc = new double[array.length];
		
		acc[0] = array[0];
		for(int i=1; i<array.length; i++){
			acc[i] = acc[i-1] + array[i];
		}
		
		return acc;
	}
	
	/**
	 * Selects n indexes given an array of weights
	 * 
	 * @param weights Array with weights of selection
	 * @param n Number of indices to select
	 * @param seed Random numbers seed
	 * 
	 * @return Array with selected indexes
	 */
	public static int [] selectNBasedOnWeights(double [] weights, int n, long seed){
		double [] acc = new double[weights.length];
		
		int [] selectedArray = new int[n];
		int selected = 0;
		
		Random rand = new Random(seed);
		
		acc = getAccumulatedArray(weights);
		
		double r;
		do{
			r = rand.nextDouble() * acc[acc.length-1];
			
			for(int i=0; i<acc.length; i++){
				if(r <= acc[i]){
					selectedArray[selected] = i;
					weights[i] = 0;
					acc = getAccumulatedArray(weights);
				}
			}
		}while(selected < n);
		
		return selectedArray;
	}

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
	 * Prints int matrix
	 * 
	 * @param matrix Matrix of int values
	 */
	public static void printMatrix(int [][] matrix){
		for(int i=0; i<matrix.length; i++){
			System.out.println(Arrays.toString(matrix[i]));
		}
	}
	
	/**
	 * Prints byte matrix
	 * 
	 * @param matrix Matrix of byte values
	 */
	public static void printMatrix(byte [][] matrix){
		for(int i=0; i<matrix.length; i++){
			System.out.println(Arrays.toString(matrix[i]));
		}
	}
	
	/**
	 * Prints double matrix
	 * 
	 * @param matrix Matrix of double values
	 */
	public static void printMatrix(double [][] matrix){
		for(int i=0; i<matrix.length; i++){
			System.out.println(Arrays.toString(matrix[i]));
		}
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
	 * @param appearances Int array with the appearances of each label
	 * 
	 * @return Frequency of the number of appearances
	 */
	public static double [] calculateFrequencies(int [] appearances){
		int total = 0;
		for(int i=0; i<appearances.length; i++){
			total += appearances[i];
		}
		
		double [] frequency = new double[appearances.length];
		for(int i=0; i<appearances.length; i++){
			frequency[i] = (double)appearances[i] / total;
		}
		
		return frequency;
	}
	
	/**
	 * Calculate the accumulated frequency given a double array
	 * 
	 * @param appearances Double array with the weight of each label
	 * 
	 * @return Frequency of the number of appearances
	 */
	public static double [] calculateFrequencies(double [] appearances){
		double total = 0;
		for(int i=0; i<appearances.length; i++){
			total += appearances[i];
		}
		
		double [] frequency = new double[appearances.length];
		for(int i=0; i<appearances.length; i++){
			frequency[i] = (double)appearances[i] / total;
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
	 * Calculate the number of expected votes of each label given a weights vector and the min and max number of votes per label
	 * 
	 * @param weights Weights representing the importance of the label
	 * @param totalVotes Total number of votes to share
	 * @param maxVotes Maximum number of votes per label
	 * @param minVotes Minimum number of votes per label
	 * @param seed Seed for random numbers
	 * @return
	 */
	public static int [] calculateExpectedVotes(double[] weights, int totalVotes, int maxVotes, int minVotes, int seed){	
		int numLabels = weights.length;
		
		int [] expectedVotes = new int[numLabels];
		
		int fix = 0;
		int maxReached = 0;
		
		for(int i=0; i<numLabels; i++){
			expectedVotes[i] = minVotes;
			totalVotes -= minVotes;
		}
		
		for(int i=0; i<numLabels; i++){
			int share = (int)Math.round(weights[i]*totalVotes);
			
			expectedVotes[i] += share;
			
			if(expectedVotes[i] > totalVotes){
				fix += (expectedVotes[i] - maxVotes);
				expectedVotes[i] = maxVotes;
			}
		}
		
		if(fix > 0){
			if(fix > (numLabels-maxReached)){
				for(int i=0; i<numLabels; i++){
					if(expectedVotes[i] < maxVotes){
						expectedVotes[i]++;
						if(expectedVotes[i] == maxVotes){
							maxReached++;
						}
					}
				}
				fix -= numLabels;
			}
			else{
				byte [] selected = new byte[numLabels];
				for(int i=0; i<numLabels; i++){
					if(expectedVotes[i] == maxVotes){
						selected[i] = 1;
					}
					
					while(fix > 0){
						int index = getMinIndex(expectedVotes, selected, seed);
						selected[index]++;
						expectedVotes[index]++;
					}
				}
			}
		}
		
		return expectedVotes;
	}	
	
	/**
	 * Looks if an array contains any value of the second array
	 * 
	 * @param array Array where to find the value
	 * @param from Array with the values to find
	 * @return
	 */
	public static boolean containsAnyFrom(int [] array, int [] from){
		for(int i=0; i<array.length; i++){
			if(contains(from, array[i])){
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
	 * Weighted hamming distance between two arrays 
	 * 
	 * @param a1 Array 1
	 * @param a2 Array 2
	 * @param weights Array with weights to calculate the hamming distance. The greater is the weight, the greater is the distance for a given label
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
	
	public static double distance(MultipBinArrayIndividual ind1, MultipBinArrayIndividual ind2, double [] weights, double ratioTrain){
		double hd = hammingDistance(ind1.getGenotype(), ind2.getGenotype(), weights);
		
		return hd;
		
		/*
		double alpha = 1-ratioTrain;
		alpha = alpha/(1+alpha);
		
		int diffData = 1;
		if(ind1.getSubpop() == ind2.getSubpop()) {
			diffData = 0;
		}
		
		return alpha*diffData + (1-alpha)*hd;
		*/
	}
	
	/**
	 * Get the index with the max value of an array
	 * 
	 * @param array Double array
	 * 
	 * @return Index of the maximum value of the array
	 */
	public static int getMaxIndex(double [] array){
		double maxValue = array[0];
		int maxIndex = 0;
		
		for(int i=1; i<array.length; i++){
			if(array[i] > maxValue){
				maxValue = array[i];
				maxIndex = i;
			}
		}
		
		return maxIndex;
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
	
}
