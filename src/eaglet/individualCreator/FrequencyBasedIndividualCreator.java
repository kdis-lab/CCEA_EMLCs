package eaglet.individualCreator;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import eaglet.utils.Utils;
import net.sf.jclec.IIndividual;

/**
 * Class implementing the creation of individuals
 * 
 * @author Jose M. Moyano
 *
 */
public class FrequencyBasedIndividualCreator extends EagletIndividualCreator {

	/**
	 * Serialization constant
	 */
	private static final long serialVersionUID = -3389888931465909537L;
	
	/**
	 *  Array with the frequencies of each label in the dataset
	 */
	private int [] appearances;
	
	/**
	 * Constructor
	 */
	public FrequencyBasedIndividualCreator()
	{
		super();
	}
	
	/**
	 * Sets the frequencies
	 * 
	 * @param frequencies Array with frequencies
	 */
	public void setAppearances(int [] appearances)
	{
		this.appearances = appearances;
	}
	
	/**
	 * Create an individual given a genotype
	 * 
	 * @param genotype Genotype as byte array
	 */
	protected void createNext(byte [] genotype)
	{
		createdBuffer.add(species.createIndividual(genotype, p));
	}

	@Override
	public List<IIndividual> provide(int numberOfIndividuals) 
	{
		// Set numberOfIndividuals
		this.numberOfIndividuals = numberOfIndividuals;
		// Result list
		createdBuffer = new ArrayList<IIndividual> (numberOfIndividuals);
		// Prepare process
		prepareCreation();
		
		createdBuffer = generateIndividuals(numberOfIndividuals);
		
		replaceRepeatedByRandomIndividuals();
		
		// Returns result
		return createdBuffer;
	}
	
	/**
	 * Generate individuals.
	 * 
	 * @param numberOfIndividuals Number of individuals
	 * @return List of generated IIndividuals
	 */
	public List<IIndividual> generateIndividuals(int numberOfIndividuals){	
		//Total number of active bits to share
		int toShare = k * numberOfIndividuals;
		
		//Number of bits shared to each label
		int [] shared = new int[numLabels];
		
		//Indicates if each label has reached the max number of active bits
		boolean [] maxsReached = new boolean[numLabels];
		
		//Share at least one active bit to each label
		for(int l=0; l<numLabels; l++){
			shared[l] = 1;
			toShare--;
			maxsReached[l] = false;
		}
		
		//Indicates if a max has been reached in the current iteration
		boolean iMaxReached;
		
		//Ratio of corresponding active bits per label
		double [] ratio;
		int corresponding;
		int remaining = toShare;
		
		do{
			iMaxReached = false;
			
			//Remaining bits to this iteration
			remaining = toShare;
			//Re-calculate ratio not taking into account maxsReached labels
			ratio = recalculateRatio(appearances, maxsReached);
			
			//Share the rest of active bits based on the frequency
			for(int l=0; l<numLabels; l++){
				if(maxsReached[l] == false){
					//Calculate corresponding number of active bits to label l
					corresponding = (int) Math.round(ratio[l] * remaining);
					
					if((shared[l] + corresponding) > numberOfIndividuals){
						//If max of this label is reached, only add until max
						//corresponding = shared[l] - corresponding;
						corresponding = numberOfIndividuals - shared[l];
						
						shared[l] += corresponding;
						toShare -= corresponding;
						
						//Max has been reached at label l
						iMaxReached = true;
						maxsReached[l] = true;
					}
					else{
						shared[l] += corresponding;
						toShare -= corresponding;
					}
				}
			}
			
		}while(iMaxReached == true);
		
		
		//Adjust if there has been more or less shared active bits
		if(toShare > 0){
			shared = addToMinority(shared, toShare);
		}
		else if(toShare < 0){
			shared = removeFromMajority(shared, Math.abs(toShare));
		}
		
		//Each individual is now represented as an ArrayList of k integers
		HashMap<Integer, ArrayList<Integer>> individuals = new HashMap<Integer, ArrayList<Integer>>();
		for(int i=0; i<numberOfIndividuals; i++){
			individuals.put(i, new ArrayList<Integer>());
		}
				
		int [] v;
		int i;
		ArrayList<Integer> currentInd;
		
		//Labels sorted by descending order of appearance. This is useful to share the active bits wit less problems.
		int [] labelsByAppearanceOrder = orderLabelByFrequency(appearances);
		
		for(int l=0; l<numLabels; l++){
			int label = labelsByAppearanceOrder[l];
			
			v = Utils.shuffledArray(numberOfIndividuals, randgen.choose(100));
			i = 0;

			do{
				currentInd = individuals.get(v[i]);
				if(currentInd.size() < k){
					currentInd.add(label);
					individuals.put(v[i], currentInd);
					shared[label]--;
				}
				
				i++;
			}while((shared[label] > 0) && (i<numberOfIndividuals));
			
			if(shared[label] > 0){
				do{
					ArrayList<Integer> ind1 = null;
					ArrayList<Integer> ind2 = null;
					
					//Look for an individual with active < k
					for(int j=0; j<numberOfIndividuals; j++){
						currentInd = individuals.get(v[j]);
						if(currentInd.size() < k){
							ind1 = currentInd;
							break;
						}
					}
					
					//Look for an individual with active = k and not label
					for(int j=0; j<numberOfIndividuals; j++){
						currentInd = individuals.get(v[j]);
						if((currentInd.size() == k) && (!currentInd.contains(label))){
							ind2 = currentInd;
							break;
						}
					}

					if(ind2 == null) {
						int r;
						do {
							r = randgen.choose(0, numLabels);
						}while(ind1.contains(r));
						
						ind1.add(r);
					}
					else {
						//Swap a label from ind2 to ind1 and add current label to ind2
						for(int j=0; j<ind2.size(); j++){
							int [] v2 = Utils.shuffledArray(ind2.size(), randgen.choose(100));
							if(!ind1.contains(ind2.get(v2[j]))){
								ind1.add(ind2.get(v2[j]));
								ind2.remove(v2[j]);
								ind2.add(label);
								shared[label]--;
								break;
							}
						}
					}
				}while(shared[label] > 0);	
			}
		}
		
		
		for (createdCounter=0; createdCounter<numberOfIndividuals; createdCounter++) {
			createNext(listToGenotype(individuals.get(createdCounter), numLabels));
		}
		
		return createdBuffer;
	}
	
	/**
	 * Recalculate the ratio of appearances without excluded labels
	 * 
	 * @param appearances Appearances of each label in the dataset
	 * @param excluded Excluded labels for the calculation of the ratio of appearance
	 * @return
	 */
	double [] recalculateRatio(int [] appearances, boolean [] excluded){
		double total = 0;
		double [] recalculatedRatio = new double [appearances.length];
		
		for(int i=0; i<appearances.length; i++){
			if(!excluded[i]){
				total += appearances[i];
			}
		}
		
		for(int i=0; i<appearances.length; i++){
			if(!excluded[i]){
				recalculatedRatio[i] = appearances[i] / total;
			}
			else{
				recalculatedRatio[i] = 0;
			}
		}
		
		return recalculatedRatio;
	}
	
	/**
	 * Given a list of labels, calculate the genotype of the individual
	 * 
	 * @param list List of integer defining the labels
	 * @param n Number of labels
	 * 
	 * @return Genotype with the labels included in list
	 */
	public byte [] listToGenotype(ArrayList<Integer> list, int n){
		byte [] genotype = new byte[n];
		
		for(int i : list){
			genotype[i] = 1;
		}
		
		return genotype;
	}
	
	/**
	 * Add n to minority labels
	 * 
	 * @param a array
	 * @param n times to share
	 * 
	 * @return array with minority values incremented
	 */
	int [] addToMinority(int [] a, int n){
		int [] array = new int[a.length];
		System.arraycopy(a, 0, array, 0, a.length);
		
		int min;
		ArrayList<Integer> minIndices = new ArrayList<Integer>();
		do{
			minIndices.clear();
			
			min = array[0];
			minIndices.add(0);
			for(int l=1; l<numLabels; l++){
				if(array[l] < min){
					min = array[l];
					minIndices.clear();
					minIndices.add(l);
				}
				else if(array[l] == min){
					minIndices.add(l);
				}
			}
			
			if(minIndices.size() <= n){
				for(int i : minIndices){
					array[i]++;
					n--;
				}
			}
			else{
				int r;
				for(int i=0; i<n; i++){
					r = randgen.choose(0, minIndices.size());
					array[minIndices.get(r)] ++;
					minIndices.remove((Integer) r);
				}
				n=0;
			}
		}while(n > 0);
		
		return array;
	}
	
	/**
	 * Remove n from majority labels
	 * 
	 * @param a array
	 * @param n times to share
	 * 
	 * @return array with majority values decremented
	 */
	int [] removeFromMajority(int [] a, int n){
		int [] array = a.clone();
		
		int max;
		ArrayList<Integer> maxIndices = new ArrayList<Integer>();
		do{
			maxIndices.clear();
			
			max = array[0];
			maxIndices.add(0);
			for(int l=1; l<numLabels; l++){
				if(array[l] > max){
					max = array[l];
					maxIndices.clear();
					maxIndices.add(l);
				}
				else if(array[l] == max){
					maxIndices.add(l);
				}
			}
			
			if(maxIndices.size() <= n){
				for(int i : maxIndices){
					array[i]--;
					n--;
				}
			}
			else{
				int r;
				for(int i=0; i<n; i++){
					r = randgen.choose(0, maxIndices.size());
					array[maxIndices.get(r)]--;
					maxIndices.remove((Integer) r);
				}
				n=0;
			}
		}while(n > 0);
		
		return array;
	}
	
	/**
	 * Order the label indices by their frequency in descending order
	 * 
	 * @param appearances Appearances of each label in the dataset
	 * @return Int array with the label indices in descending order of frequency
	 */
	int [] orderLabelByFrequency(int [] appearances){
		int [] appCopy = appearances.clone();
		
		int [] order = new int[appearances.length];
		
		int max, maxIndex=-1;
		
		for(int i=0; i<appearances.length; i++){
			max = Integer.MIN_VALUE;
			for(int j=0; j<appearances.length; j++){
				if(appCopy[j] > max){
					max = appCopy[j];
					maxIndex = j;
				}
			}
			
			order[i] = maxIndex;
			appCopy[maxIndex] = Integer.MIN_VALUE;
		}
		
		return order;
	}

	
}
