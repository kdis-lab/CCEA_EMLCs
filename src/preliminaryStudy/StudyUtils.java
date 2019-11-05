package preliminaryStudy;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.DoubleStream;

import coeaglet.utils.Utils;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.measure.*;
import net.sf.jclec.IIndividual;
import net.sf.jclec.listind.MultipListGenotype;
import net.sf.jclec.listind.MultipListIndividual;
import net.sf.jclec.util.random.IRandGen;

public class StudyUtils {

	/**
	 * Get the header of the csv file
	 * 
	 * @param parameters String with the parameters of the execution
	 * @return String with the header
	 */
    public static String getCSVHeader(String parameters) {
    	String header = new String();
    	
    	header += parameters;
    	
    	header += new HammingLoss().getName() + "; ";
    	header += new ModHammingLoss().getName() + "; ";
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
    	//header += new MacroAUC(2).getName() + "; ";
    	
    	//header += "HierarchicalLoss" + "; ";

        return header;
    }

    
    /**
	 * Generate individuals for a given subpop
	 * 
	 * @param numberOfIndividuals Number of individuals
	 * @return List of generated IIndividuals
	 */
    
    /**
     * Generate individuals for a given subpop
     * 
     * @param numberOfIndividuals Number of individuals in the subpopulation
     * @param subpop Index of subpopulation
     * @param appearances Appearances of each label in the corresponding training data
     * @param aMin Minimum number of times that each label appears in the subpopulation
     * @param k Number of active labels in each member
     * @param randgen Random numbers generator
     * @return List of generated individuals
     */
	public static List<MultipListIndividual> generateIndividuals(int numberOfIndividuals, int subpop, int [] appearances, int aMin, int k, IRandGen randgen){		
		//List of indiviuals
		List<MultipListIndividual> inds = new ArrayList<MultipListIndividual>(numberOfIndividuals);
		
		//Calculate initial weights given appearances
		double [] weights = new double[appearances.length];
		double sumAppearances = Arrays.stream(appearances).sum();		
		double [] baseVotes = new double[appearances.length];
		
		for(int i=0; i<weights.length; i++) {
			//Expected votes per label is the ratio of appearance of the label divided by the number of bits to share
			baseVotes[i] = (appearances[i] / sumAppearances) * (numberOfIndividuals*k);
			
			//If the expected votes is lower than the minimum, set the minimum
			if(baseVotes[i] < aMin) {
				weights[i] = aMin;
			}
			//The expected votes are upper bounded by the number of individuals
			else if(baseVotes[i] > numberOfIndividuals) {
				weights[i] = numberOfIndividuals;
			}
			else {
				weights[i] = baseVotes[i];
			}
		}
		
		//Calculate probabilities given weights
		double [] prob = new double[appearances.length];
		double sumWeights = DoubleStream.of(weights).sum();
		for(int i=0; i<weights.length; i++) {
			prob[i] = weights[i] / sumWeights;
		}
		
		//Create individuals
		ArrayList<Integer> list = new ArrayList<Integer>(k);
		int index;
		double [] probCopy;
		
		//Until the desired number of individuals is created
		while(inds.size() < numberOfIndividuals) {
			//Clear list of genotype
			list.clear();
			
			//Copy array of probabilities
			probCopy = prob.clone();

			while(list.size() < k) {
				//Select random index (based on probs) to add to the individual
				index = Utils.probabilitySelectIndex(probCopy, randgen);
				
				//Add index to the individual
				list.add(index);
				
				//Give 0 probability to the same index to be selected
				probCopy[index] = 0;
			}
			
			//Sort individual
			Collections.sort(list);
			
			//Add individual to the list if it still does not exist
			MultipListIndividual newInd = new MultipListIndividual(new MultipListGenotype(subpop, new ArrayList<Integer>(list)));
			if(!contains(inds, newInd))
			{	
				inds.add(newInd);
			}
		}
		
		//Check if all labels appear at least aMin times
		int [] labelAppearances = getAppearancesLabels(inds, appearances.length);
		
		for(int i=0; i<labelAppearances.length; i++) {
			//If a label appear less than aMin
			while(labelAppearances[i] < aMin) {
				//Get an individual including the label that most appear in the population
				IIndividual maxLabelInd = getMaxLabelIndividual(inds, labelAppearances, aMin, randgen);
				int p = ((MultipListIndividual)maxLabelInd).getGenotype().subpop;
				ArrayList<Integer> gen = ((MultipListIndividual)maxLabelInd).getGenotype().genotype;
				
				//Get index of max label
				int maxIndex = 0;
				for(int j=1; j<gen.size(); j++) {
					if(labelAppearances[gen.get(j)] > labelAppearances[maxIndex]) {
						maxIndex = j;
					}
				}
				
				//Quit index of max label and add i-th label to genotype
				labelAppearances[i]++;
				labelAppearances[gen.get(maxIndex)]--;
				gen.remove(maxIndex);
				gen.add(i);
				Collections.sort(gen);
				
				//Remove old individual and add new one to subpopulation
				inds.remove(maxLabelInd);
				inds.add(new MultipListIndividual(new MultipListGenotype(p, gen)));
			}
		}
		
//		System.out.println("labelAppearances: " + Arrays.toString(labelAppearances));
		
		return inds;
	}
	
	/**
	 * Check if a MultipListIndividual exists in a population
	 * 
	 * @param pop Population
	 * @param ind MultipListIndividual
	 * 
	 * @return true if ind exists in pop, and false if not.
	 */
	public static boolean contains(List<MultipListIndividual> pop, MultipListIndividual ind) {
		for(IIndividual pInd : pop) {
			if(ind.equals(pInd)){
				return true;
			}
		}
		
		return false;
	}
	
	/**
	 * Get an individual with maximum label (label that most appear), but not critical label (labels appearing <= aMin)
	 * 
	 * @param inds List of individuals
	 * @param appearances Appearances of each label in the population
	 * @param aMin Minimum appearances for each label
	 * @return Individual with max label
	 */
	protected static MultipListIndividual getMaxLabelIndividual(List<MultipListIndividual> inds, int[] appearances, int aMin, IRandGen randgen) {
		ArrayList<Integer> maxLabels = new ArrayList<Integer>();
		ArrayList<Integer> criticalLabels = new ArrayList<Integer>();
		
		maxLabels.add(0);
		for(int i=1; i<appearances.length; i++) {
			if(appearances[i] > appearances[maxLabels.get(0)]) {
				maxLabels.clear();
				maxLabels.add(i);
			}
			else if(appearances[i] == appearances[maxLabels.get(0)]) {
				maxLabels.add(i);
			}
			else if(appearances[i] <= aMin) {
				criticalLabels.add(i);
			}
		}
		
		int maxLabel = maxLabels.get(randgen.choose(0, maxLabels.size()));
		
		List<MultipListIndividual> indsMaxLabel = new ArrayList<MultipListIndividual>();
		//Add individual with max label and not critical label
		for(MultipListIndividual ind : inds) {
			if(((MultipListIndividual)ind).getGenotype().genotype.contains(maxLabel) && !containsCriticalLabel((MultipListIndividual)ind, criticalLabels)) {
				indsMaxLabel.add(ind);
			}
		}
		
		//Return one of these individuals
		return indsMaxLabel.get(randgen.choose(0, indsMaxLabel.size()));
	}
	
	/**
	 * Check if a given individual contains a critical label
	 * 
	 * @param ind Individual
	 * @param criticalLabels Set of critical labels
	 * @return true if ind contains any of the critical labels and false otherwise
	 */
	protected static boolean containsCriticalLabel(MultipListIndividual ind, List<Integer> criticalLabels) {
		for(Integer g : ind.getGenotype().genotype) {
			if(criticalLabels.contains(g)) {
				return true;
			}
		}
		
		return false;
	}
	
	/**
	 * Get the appearances of each label in the population
	 * 
	 * @param inds Population of individuals
	 * @return Appearances of each label in this population
	 */
	protected static int [] getAppearancesLabels(List<MultipListIndividual> inds, int maxInt) {
		int [] appearances = new int[maxInt];
		
		for(MultipListIndividual ind : inds) {
			for(int g : (ind).getGenotype().genotype) {
				appearances[g]++;
			}
		}
		
		return appearances;
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
                measures.add(new ModHammingLoss());
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
                //measures.add(new MacroAUC(numOfLabels));
               // measures.add(new LogLoss());
            }
            // add hierarchical measures if applicable
            if (mlTestData.getLabelsMetaData().isHierarchy()) {
                measures.add(new HierarchicalLoss(mlTestData));
            }
        } catch (Exception ex) {
//            Logger.getLogger(Evaluator.class.getName()).log(Level.SEVERE, null, ex);
        	ex.printStackTrace();
        }

        return measures;
    }
}
