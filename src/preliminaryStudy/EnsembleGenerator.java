package preliminaryStudy;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;
import java.util.Random;

import eaglet.utils.Utils;
import mulan.classifier.MultiLabelLearner;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.measure.ExampleBasedFMeasure;
import mulan.evaluation.measure.Measure;
import net.sf.jclec.IIndividual;
import net.sf.jclec.binarray.MultipBinArrayIndividual;
import net.sf.jclec.fitness.SimpleValueFitness;
import preliminaryStudy.algorithm.EnsembleMLC_Multip;

/** 
 * @author Jose M. Moyano
 *
 */
public class EnsembleGenerator {

	/**
	 * Generate an ensemble (select the members) and build it given the individuals
	 * 
	 * @param individuals List of possible individuals for the ensemble
	 * @param mlData Train dataset
	 * @param n Number of individuals in the ensemble
	 * @param expectedVotes Expected number of votes for each label in the ensemble
	 * @param beta Value to give more importance to performance or diversity
	 * @return
	 */
	public EnsembleMLC_Multip generateAndBuildEnsemble(List<List<MultipBinArrayIndividual>> individuals, MultiLabelInstances mlData, int n, double beta,
			MultiLabelLearner learner, Hashtable<String, MultiLabelLearner> tableClassifiers, boolean prune, int seed) {
		List<MultipBinArrayIndividual> ensembleMembers = null;
		EnsembleMLC_Multip ensemble = null;
		List<MultipBinArrayIndividual> allInds = new ArrayList<MultipBinArrayIndividual>();
		
		int nSubpops = 0;
		
		for(List<MultipBinArrayIndividual> subpop : individuals) {
			allInds.addAll(subpop);
			nSubpops++;
		}
		
		try {
			ensembleMembers = selectEnsembleMembers(allInds, n, beta, 3, nSubpops, seed);
			if(prune) {
				ensembleMembers = pruneEnsemble(ensembleMembers, mlData, learner, tableClassifiers);
			}
			ensemble = generateEnsemble(ensembleMembers, ensembleMembers.size(), learner, tableClassifiers);
		
			ensemble.build(mlData);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return ensemble;
	}
	
	
	public List<MultipBinArrayIndividual> pruneEnsemble(List<MultipBinArrayIndividual> members, MultiLabelInstances mlData,
			MultiLabelLearner learner, Hashtable<String, MultiLabelLearner> tableClassifiers) {
		EnsembleMLC_Multip ensemble = null;
		List<MultipBinArrayIndividual> bestMembers = new ArrayList<MultipBinArrayIndividual>(members);
		List<MultipBinArrayIndividual> copyMembers = new ArrayList<MultipBinArrayIndividual>(members);
		
		double bestFit = 0.0, currFit;
		int nWorst = (int)Math.round(members.size() * .1);
		int failed = 0;
		
		try {
			ensemble = generateEnsemble(copyMembers, copyMembers.size(), learner, tableClassifiers);
			ensemble.build(mlData);
			
			Evaluation results;     	
	     	Evaluator eval = new Evaluator();
	     	List<Measure> measures = new ArrayList<Measure>();  	
	     	measures.add(new ExampleBasedFMeasure());
	     	results = eval.evaluate(ensemble, mlData, measures);
	     	bestFit = results.getMeasures().get(0).getValue();
	     	//System.out.println("Fitness: " + bestFit);
	     	
	     	while(failed < nWorst) {
	     		copyMembers.remove(copyMembers.size()-1);
	     		ensemble = generateEnsemble(copyMembers, copyMembers.size(), learner, tableClassifiers);
				ensemble.build(mlData);
		     	results = eval.evaluate(ensemble, mlData, measures);
		     	currFit = results.getMeasures().get(0).getValue();
	     		
		     	if(currFit > bestFit) {
		     		bestFit = currFit;
		     		bestMembers = new ArrayList<MultipBinArrayIndividual>(ensemble.getEnsembleInds());
		     		failed = 0;
		     		//System.out.println("Improved: " + bestFit);
		     	}
		     	else {
		     		failed++;
		     		//System.out.println("Failed. ; " + currFit);
		     	}
	     	}
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		
		return bestMembers;
	}
	
	/**
	 * Select the members that form the ensemble given the individuals and the expected votes per label
	 * 
	 * @param individuals List of possible individuals for the ensemble
	 * @param n Number of members in the ensemble
	 * @param expectedVotes Expected votes per label in the ensemble
	 * @param beta Value to give more importance to performance or to diversity
	 * @return List of individuals selected to form the ensemble
	 */
	public List<MultipBinArrayIndividual> selectEnsembleMembers(List<MultipBinArrayIndividual> individuals, int n, double beta, int k, int nSubpops, int seed){
		BettersSelector2 bselector = new BettersSelector2();
		int numberLabels = individuals.get(0).getGenotype().length;
		
		int [] expectedVotes = Utils.spreadVotesEvenly(numberLabels, (int)Math.round((10.0/k)*numberLabels)*k, new Random(1).nextInt(100));
		
		//Copy of the expectedVotes array
		int [] expectedVotesCopy = new int[numberLabels];
		System.arraycopy(expectedVotes, 0, expectedVotesCopy, 0, numberLabels);
		
		//Weights for each label
		//Spread votes evenly
		double [] weights = new double[numberLabels];
		for(int j=0; j<numberLabels; j++){
			weights[j] = (double)1 / numberLabels;
		}
		
		
		byte [][] EnsembleMatrix = new byte[n][numberLabels];
		
		List<MultipBinArrayIndividual> members = new ArrayList<MultipBinArrayIndividual>();
		
		List<MultipBinArrayIndividual> indsCopy = individuals; 

		//Sort individuals by fitness
		indsCopy = bselector.selectMultip(indsCopy, indsCopy.size());
		
		//Add first individual to ensemble members and remove from list
		members.add((MultipBinArrayIndividual)indsCopy.get(0));
		System.arraycopy(((MultipBinArrayIndividual)indsCopy.get(0)).getGenotype(), 0, EnsembleMatrix[0], 0, numberLabels);
		indsCopy.remove(0);
		
		//For each remaining individual, compute its new fitness as a combination of its fitness and the distance to the ensemble
		int currentEnsembleSize = 1;
		
		double [] updatedFitnesses;
		do{
			//Calculate weights with current expected votes array
			weights = Utils.calculateFrequencies(expectedVotesCopy);
			updatedFitnesses = new double[indsCopy.size()];
			
			//Update fitness for all individuals
			for(int i=0; i<indsCopy.size(); i++){
				updatedFitnesses[i] = beta * distanceToEnsemble((MultipBinArrayIndividual)indsCopy.get(i), members, currentEnsembleSize, weights) + (1-beta)*((SimpleValueFitness)indsCopy.get(i).getFitness()).getValue();
			}
			
			//Get best individual with updated fitness
			int maxIndex = Utils.getMaxIndex(updatedFitnesses, seed);
			
			//Add individual to ensemble members
			members.add((MultipBinArrayIndividual)indsCopy.get(maxIndex));
			//Update expectedVotesCopy to then recalculate weights (keep a minumum of 1)
			IIndividual currInd = indsCopy.get(maxIndex);
			byte [] currGen = ((MultipBinArrayIndividual)currInd).getGenotype();
			for(int i=0; i<currGen.length; i++){
				if(currGen[i] == 1){
					if(expectedVotesCopy[i] > 1){
						expectedVotesCopy[i]--;
					}
				}
			}
			
			System.arraycopy(((MultipBinArrayIndividual)indsCopy.get(maxIndex)).getGenotype(), 0, EnsembleMatrix[currentEnsembleSize], 0, numberLabels);
			//Remove individual from list
			indsCopy.remove(maxIndex);
						
			currentEnsembleSize++;
		}while(currentEnsembleSize < n);
		
		//Ensure all labels are taken into account in the ensemble
		ArrayList<Integer> noVotesLabels = new ArrayList<Integer>();
		do{
			noVotesLabels.clear();
			int [] votesPerLabel = Utils.calculateVotesPerLabel(individualsToEnsembleMatrix(members, numberLabels));

			for(int i=0; i<votesPerLabel.length; i++){
				if(votesPerLabel[i] == 0){
					noVotesLabels.add(i);
				}
			}

			if(noVotesLabels.size() > 0){
				//System.out.println("\n -> " + noVotesLabels.toString());
				
				Random rand = new Random(1);
				
				weights = Utils.calculateFrequencies(expectedVotes);
				int r = rand.nextInt(noVotesLabels.size());

				int currentLabel = noVotesLabels.get(r);

				//Remove the worst individual of the most voted label (if do not remove other label appearing only once
				MultipBinArrayIndividual worstIndByLabel = getWorstIndividualByLabel(members, Utils.getMaxIndex(votesPerLabel, rand.nextInt(100)));
				members.remove(worstIndByLabel);
				
				//Add the individual including label noVotesLabels[r] that better matches with the ensemble
				List<MultipBinArrayIndividual> candidates = getIndividualsWithLabel(indsCopy, currentLabel);
				
				/*
				if(candidates.size() <= 0) {
					System.out.println("Adding new individual");
					candidates.add(new MultipBinArrayIndividual(createIndWithLabel(numberLabels, currentLabel, k, rand.nextInt()%100), rand.nextInt()%nSubpops));
					MemberEvaluator eval = new MemberEvaluator();
					eval.evaluate(candidates.get(0));
				}
				System.out.println(candidates.size());
				*/
				
				
				double [] candidatesFitness = new double[candidates.size()];
				
				for(int i=0; i<candidates.size(); i++){
					candidatesFitness[i] = beta * distanceToEnsemble((MultipBinArrayIndividual)candidates.get(i), members, members.size(), weights) + (1-beta)*((SimpleValueFitness)candidates.get(i).getFitness()).getValue();
				}
				
				double maxFitness = candidatesFitness[0];
				int maxFitnessIndex = 0;
				for(int i=1; i<candidatesFitness.length; i++){
					if(candidatesFitness[i] > maxFitness){
						maxFitness = candidatesFitness[i];
						maxFitnessIndex = i;
					}
				}
				
				members.add((MultipBinArrayIndividual)candidates.get(maxFitnessIndex));
				indsCopy.remove(candidates.get(maxFitnessIndex));
				
				//Re-include the removed indivudual in the indsCopy set.
				indsCopy.add(worstIndByLabel);
			}
		}while(noVotesLabels.size() > 0);
		
		return members;
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

	/**
	 * Generate an ensemble given the list of individuals
	 * 
	 * @param members Members of the ensemble
	 * @return Ensemble generated
	 */
	public EnsembleMLC_Multip generateEnsemble(List<MultipBinArrayIndividual> members, int numClassifiers,
			MultiLabelLearner learner, Hashtable<String, MultiLabelLearner> tableClassifiers){
		EnsembleMLC_Multip ensemble = new EnsembleMLC_Multip(members, learner, numClassifiers, tableClassifiers);
		ensemble.setThreshold(0.5);
		ensemble.setSeed(1);
		return ensemble;
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
	public double distanceToEnsemble(MultipBinArrayIndividual ind, List<MultipBinArrayIndividual> ensemble, int ensembleSize, double [] weights){
		double distance = 0;
		
		for(int i=0; i<ensembleSize; i++){
			distance += Utils.distance(ind, ensemble.get(i), weights);
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
	public byte [][] individualsToEnsembleMatrix(List<MultipBinArrayIndividual> individuals, int numberLabels){
		byte [][] EnsembleMatrix = new byte[individuals.size()][numberLabels];
		
		for(int i=0; i<individuals.size(); i++){
			System.arraycopy(((MultipBinArrayIndividual)individuals.get(i)).getGenotype(), 0, EnsembleMatrix[i], 0, numberLabels);
		}
		
		return EnsembleMatrix;
	}
	
	/**
	 * Obtain all individuals containing a given label
	 * 
	 * @param indsCopy List of individuals
	 * @param label Label
	 * @return
	 */
	public List<MultipBinArrayIndividual> getIndividualsWithLabel(List<MultipBinArrayIndividual> indsCopy, int label){
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
	
	/**
	 * Get the individual with less predictive performance from those containing a given label
	 * 
	 * @param individuals List of individuals
	 * @param label Given label
	 * @return Worst individual with the label
	 */
	public MultipBinArrayIndividual getWorstIndividualByLabel(List<MultipBinArrayIndividual> individuals, int label){
		List<MultipBinArrayIndividual> candidates = new ArrayList<MultipBinArrayIndividual>();
		
		for(int i=0; i<individuals.size(); i++){
			if((individuals.get(i)).getGenotype()[label] == 1){
				candidates.add(individuals.get(i));
			}
		}
		
		double minFitness = Double.MAX_VALUE;
		MultipBinArrayIndividual worst = null;
		
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
	 * Indicates if an individual has a critical label.
	 * It is a critical label if it only appears once in the population
	 * 
	 * @param ind Individual to check
	 * @param list List of individuals
	 * @return True if any of the labels in ind is critical
	 */
	public boolean hasCriticalLabel(MultipBinArrayIndividual ind, List<MultipBinArrayIndividual> list){
		int numberLabels = ind.getGenotype().length;
		
		int [] votesPerLabel = Utils.calculateVotesPerLabel(individualsToEnsembleMatrix(list, numberLabels));
		
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
}
