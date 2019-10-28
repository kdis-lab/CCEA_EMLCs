package coeaglet.utils;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.DoubleStream;

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import net.sf.jclec.IIndividual;
import net.sf.jclec.listind.MultipListIndividual;
import net.sf.jclec.util.random.IRandGen;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Class implementing some utilities
 * 
 * @author Jose M. Moyano
 *
 */
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
	 * Types of communication between subpopulations
	 *	no: No communication between subpopulations during the evolution; only at the end the ensemble is generated
	 *	exchange: Promising individuals are copied to other subpopulations, while useless individuals are removed
	 *	operators: Specific genetic operators between subpopulations are used to share information.
	 */
	public enum CommunicationType{
		no, exchange, operators,
	};
	
	/**
	 * Partition data into train and validation sets
	 * 
	 * @param mlData Full data
	 * @param samplingTechnique Technique for selecting the data
	 * @return Sampled dataset
	 */
	public static MultiLabelInstances sampleData(MultiLabelInstances mlData, double ratio, IRandGen randgen){
		MultiLabelInstances newMLData = null;
		Instances data, newData;
		
		//Create new empty dataset
		data = mlData.getDataSet();
		newData = new Instances(data);
		newData.removeAll(newData);
		
		//Get shuffle array of indexes
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
		
		//Number of instances to keep
		int limit = (int)Math.round(data.numInstances() * ratio); 

		//Add corresponding random instances to new data
		for(int i=0; i<limit; i++) {
			newData.add(data.get(indexes[i]));
		}
		
		//Generate new multi-label dataset
		try {
			newMLData = new MultiLabelInstances(newData, mlData.getLabelsMetaData());
		} catch (InvalidDataFormatException e1) {
			e1.printStackTrace();
		}
		
		return newMLData;
	}
	
	/**
	 * Get index of maximum in array. If several maximums, return a random one
	 * 
	 * @param array Array of values to find max
	 * @param randgen Random number generator
	 * @return Index of max (or random index of one on the maxs)
	 */
	public static int getMaxIndex(double [] array, IRandGen randgen) {
		ArrayList<Integer> maxIndex = new ArrayList<Integer>();
		double maxValue = -1;
		
		for(int i=0; i<array.length; i++) {
			if(array[i] > maxValue) {
				maxIndex.clear();
				maxIndex.add(i);
				maxValue = array[i];
			}
			else if(array[i] == maxValue) {
				maxIndex.add(i);
			}
		}
		
		if(maxIndex.size() > 1) {
			return maxIndex.get(randgen.choose(0, maxIndex.size()));
		}
		else {
			return maxIndex.get(0);
		}
	}
	
	/**
	 * Get index of maximum in array. If several maximums, return a random one
	 * 
	 * @param array Array of values to find max
	 * @param randgen Random number generator
	 * @return Index of max (or random index of one on the maxs)
	 */
	public static int getMaxIndex(int [] array, IRandGen randgen) {
		ArrayList<Integer> maxIndex = new ArrayList<Integer>();
		double maxValue = -1;
		
		for(int i=0; i<array.length; i++) {
			if(array[i] > maxValue) {
				maxIndex.clear();
				maxIndex.add(i);
				maxValue = array[i];
			}
			else if(array[i] == maxValue) {
				maxIndex.add(i);
			}
		}
		
		if(maxIndex.size() > 1) {
			return maxIndex.get(randgen.choose(0, maxIndex.size()));
		}
		else {
			return maxIndex.get(0);
		}
	}
	
	/**
	 * Check if a MultipListIndividual exists in a population
	 * 
	 * @param pop Population
	 * @param ind MultipListIndividual
	 * 
	 * @return true if ind exists in pop, and false if not.
	 */
	public static boolean contains(List<IIndividual> pop, MultipListIndividual ind) {
		for(IIndividual pInd : pop) {
			if(ind.equals((MultipListIndividual)pInd)){
				return true;
			}
		}
		
		return false;
	}
	
	/**
	 * Get appearances of each label in dataset
	 * 
	 * @param mlData Multi-label dataset
	 * @return Int array with appearances of each label
	 */
	public static int [] getAppearances(MultiLabelInstances mlData){
		int [] appearances = new int[mlData.getNumLabels()];
		int [] labelIndices = mlData.getLabelIndices();
		
		for(Instance instance : mlData.getDataSet()){
			for(int label=0; label<mlData.getNumLabels(); label++){
				if(instance.value(labelIndices[label]) == 1){
					appearances[label]++;
				}
			}
		}
		
		return appearances;
	}
	
	/**
	 * This method returns a random index based on the array of probabilities
	 * The greater the value of each array position, the greater its probability to be selected.
	 * 
	 * @param array Array with probabilistic values
	 * 
	 * @return Index based on probabilities of the array
	 */
	public static int probabilitySelectIndex(double [] array, IRandGen randgen) {
		double r = randgen.uniform(0, DoubleStream.of(array).sum());
		
		for(int i=0; i<array.length; i++) {
			if(array[i] > r) {
				return i;
			}
			else {
				r -= array[i];
			}
		}

		return -1;
	}

}
