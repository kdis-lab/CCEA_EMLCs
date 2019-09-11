package eaglet.algorithm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Hashtable;
import java.util.List;

import org.apache.commons.configuration.Configuration;

import eaglet.individualCreator.EagletIndividualCreator;
import eaglet.individualCreator.FrequencyBasedIndividualCreator;
import eaglet.mutator.EagletMutator;
import eaglet.recombinator.RandomCrossover;
import eaglet.utils.Utils;
import mulan.data.InvalidDataFormatException;
import mulan.data.IterativeStratification;
import mulan.data.MultiLabelInstances;
import mulan.data.Statistics;
import net.sf.jclec.IIndividual;
import net.sf.jclec.algorithm.classic.MultiSGE;
import net.sf.jclec.binarray.MultipBinArrayIndividual;
import net.sf.jclec.fitness.SimpleValueFitness;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.transformation.LabelPowerset;
import net.sf.jclec.selector.BettersSelector;
import net.sf.jclec.util.random.IRandGen;
import weka.classifiers.trees.J48;
import weka.core.Instances;

/**
 * Class implementing the evolutionary algorithm for the optimization of MLCEnsemble
 * 
 * @author Jose M. Moyano
 *
 */
public class MLCAlgorithm extends MultiSGE {

	/**
	 * Serialization constant
	 */
	private static final long serialVersionUID = 8988750329823184523L;

	/**
	 * Full dataset train
	 */
	private MultiLabelInstances fullDatasetTrain;

	/**
	 *  Dataset to build the base classifiers 
	 */
	private MultiLabelInstances[] datasetTrain;
	
	/**
	 *  Dataset to evaluate the individuals 
	 */
	private MultiLabelInstances[] datasetValidation;
	
	/**
	 *  Dataset to evaluate the final ensemble 
	 */
	private MultiLabelInstances datasetTest;
	
	/**
	 * Number of labels of the dataset
	 */
	int numberLabels;
	
	/**
	 *  Max number of active labels in each base classifier 
	 */
	private int maxNumLabelsClassifier;
	
	/**
	 *  Number of base classifiers of the ensemble 
	 */
	private int numClassifiers;
	
	/**
	 *  Threshold for the voting process in ensemble prediction 
	 */
	private double predictionThreshold;
	
	/**
	 *  Base learner for base classifiers
	 */
	private MultiLabelLearner learner;
	
	/** 
	 * Table storing fitness of all individuals 
	 */
	private Hashtable<String, Double> tableFitness;
	
	/**
	 *  Table that stores all the base classifiers that have been built 
	 */
	private Hashtable<String, MultiLabelLearner> tableClassifiers;

	/**
	 *  Ensemble classifier 
	 */
	private EnsembleMLC ensemble;
	
	/**
	 *  3D array including phi matrices with correlations between labels.
	 *  First dimension indicates the subpopulation.
	 */
	private double [][][] phiMatrix;
	
	/**
	 *  Indicates if a validation set is used to evaluate the individuals 
	 */
	private boolean useValidationSet;
	
	/**
	 * Fitness of the best ensemble at the moment 
	 */
	public double bestFitness = 0.0;
	
	/**
	 * Best ensemble at the moment 
	 */
	private EnsembleMLC bestEnsemble;
	
	/**
	 * Fitness of the ensemble in each iteration
	 */
	public double iterEnsembleFitness;
	
	/** 
	 * Betters selector. Used in update phase 
	 */	
	private BettersSelector bettersSelector = new BettersSelector(this);
	
	/**
	 * Techniques for the validation set
	 */
	private enum ValidationSetTechnique{
		pct67, pct75, pct80, outOfBag,
	};
	ValidationSetTechnique validationSetTechnique;
	
	/**
	 * Seed for random numbers
	 */
	private long seed;
	
	/**
	 * Random numbers generator
	 */
	IRandGen randgen;
	
	/**
	 * Weights for the expected number of votes per label
	 */
	double [] weightsPerLabel;
	
	/**
	 * Number of votes expected for each label
	 */
	int [] expectedVotesPerLabel;
	
	/**
	 * More frequent labels have more chance to have more votes
	 */
	boolean weightVotesByFrequency;

	/**
	 * beta value to multiply by distance to the ensemble in member selection
	 */
	double betaMemberSelection;

	/**
	 * Constructor
	 */
	public MLCAlgorithm()
	{
		super();
		tableFitness = new Hashtable<String, Double>();
		tableClassifiers = new Hashtable<String, MultiLabelLearner>();
		bestFitness = Double.MIN_VALUE;
		
		learner = new LabelPowerset(new J48());
	}
	
	
	/**
	 * Gets the train multi-label dataset
	 * 
	 * @return Multi-label train dataset
	 */
	public MultiLabelInstances getDatasetTrain(int i)
	{
		return datasetTrain[i];
	}
	
	/**
	 * Gets the validation multi-label dataset
	 * 
	 * @return Multi-label validation dataset
	 */
	public MultiLabelInstances getDatasetValidation(int i)
	{
		return datasetValidation[i];
	}
	
	/**
	 * Gets the test multi-label dataset
	 * 
	 * @return Multi-label test dataset
	 */
	public MultiLabelInstances getDatasetTest()
	{
		return datasetTest;
	}
	
	/**
	 * Gets the full training multi-label dataset
	 * 
	 * @return Multi-label full training dataset
	 */
	public MultiLabelInstances getFullDatasetTrain()
	{
		return fullDatasetTrain;
	}
	
	/**
	 * Gets the max number of labels per classifier
	 * 
	 * @return Max number of labels per classifier
	 */
	public int getMaxNumberLabelsClassifier()
	{
		return maxNumLabelsClassifier;
	}
	
	/**
	 * Gets the ensemble classifier
	 * 
	 * @return Ensemble
	 */
	public EnsembleMLC getEnsemble()
	{
		return ensemble;
	}
	
	/**
	 * Returns if a validation set is or not used
	 * 
	 * @return True if a validation set is used and false otherwise
	 */
	public boolean getIsValidationSet()
	{
		return useValidationSet;
	}
	
	/**
	 * Get the number of evaluated individuals in the evolution
	 * 
	 * @return Number of evaluated individuals
	 */
	public int getNumberOfEvaluatedIndividuals(){
		return this.tableFitness.size();
	}
	
	/**
	 * Configure some default aspects and parameters of EME to make the configuration easier
	 * 
	 * @param configuration Configuration
	 */
	private void configureEagletDefaults(Configuration configuration) {
		//Species
		configuration.setProperty("species[@type]", "net.sf.jclec.binarray.MultipBinArrayIndividualSpecies");
		configuration.setProperty("species[@genotype-length]", "1");
		
		//Validation set (only if not provided)
		if(! configuration.containsKey("validation-set")) {
			configuration.addProperty("validation-set", "false");
		}
		
		//Evaluator (only if not provided)
		if(! configuration.containsKey("evaluator[@type]")) {
			configuration.addProperty("evaluator[@type]", "eaglet.algorithm.MLCEvaluator");
		}
		
		//Provider (only if not provided)
		if(! configuration.containsKey("provider[@type]")) {
			configuration.addProperty("provider[@type]", "eaglet.individualCreator.FrequencyBasedIndividualCreator");
		}
		
		//Randgen type (only if not provided)
		if(! configuration.containsKey("rand-gen-factory[@type]")) {
			configuration.addProperty("rand-gen-factory[@type]", "net.sf.jclec.util.random.RanecuFactory");
		}
		
		//Parents-selector (only if not provided)
		if(! configuration.containsKey("parents-selector[@type]")) {
			configuration.addProperty("parents-selector[@type]", "net.sf.jclec.selector.TournamentSelector");
		}
		if(! configuration.containsKey("parents-selector.tournament-size")) {
			configuration.addProperty("parents-selector.tournament-size", "2");
		}
		
		//Listener type (only if not provided)
		if(! configuration.containsKey("listener[@type]")) {
			configuration.addProperty("listener[@type]", "eaglet.algorithm.MLCListener");
		}
		
		//Other parameters
		if(! configuration.containsKey("weightVotesByFrequency")) {
			configuration.addProperty("weightVotesByFrequency", "false");
		}
	}
	
	
	@Override
	public void configure(Configuration configuration)
	{
		configureEagletDefaults(configuration);
		super.configure(configuration);
		
		System.out.println("Number of subpopulations: " + numSubpop);
		datasetTrain = new MultiLabelInstances[numSubpop];
		datasetValidation = new MultiLabelInstances[numSubpop];
		
		try {
			//Get seed for random numbers
			seed = configuration.getLong("rand-gen-factory[@seed]");
			
			// Read train/test datasets
			String datasetTrainFileName = configuration.getString("dataset.train-dataset");
			String datasetTestFileName = configuration.getString("dataset.test-dataset");
			String datasetXMLFileName = configuration.getString("dataset.xml");
			
			fullDatasetTrain = new MultiLabelInstances(datasetTrainFileName, datasetXMLFileName);
			datasetTest = new MultiLabelInstances(datasetTestFileName, datasetXMLFileName);

			useValidationSet = configuration.getBoolean("validation-set");
			
			String validationSetTechniqueString = configuration.getString("validation-set-type");
			//System.out.println(validationSetTechniqueString);
			switch (validationSetTechniqueString) {
			case "pct67":
				validationSetTechnique = ValidationSetTechnique.pct67;					
				break;
			case "pct75":
				validationSetTechnique = ValidationSetTechnique.pct75;					
				break;
			case "pct80":
				validationSetTechnique = ValidationSetTechnique.pct80;					
				break;
			case "outOfBag":
				validationSetTechnique = ValidationSetTechnique.outOfBag;					
				break;

			default:
				break;
			}
			
			if(useValidationSet)
			{
				for(int i=0; i<numSubpop; i++) {
					MultiLabelInstances [] m = generateValidationSet(fullDatasetTrain.clone(), validationSetTechnique);
					datasetTrain[i] = m[0];
					datasetValidation[i] = m[1];
				}
			}
			else
			{
				//Train and validation set are the same, the full set
				/*
				for(int i=0; i<numSubpop; i++) {
					datasetTrain[i] = fullDatasetTrain;
					datasetValidation[i] = fullDatasetTrain;
				}
				*/
				for(int i=0; i<numSubpop; i++) {
					seed = seed + 1;
					MultiLabelInstances [] m = generateValidationSet(fullDatasetTrain.clone(), validationSetTechnique);
					datasetTrain[i] = m[0];
					datasetValidation[i] = m[0];
					datasetValidation[i].getDataSet().addAll(m[1].getDataSet());
				}	
			}
			
			//Get number of labels
			numberLabels = fullDatasetTrain.getNumLabels();
			
			numClassifiers = configuration.getInt("number-classifiers");
			predictionThreshold = configuration.getDouble("prediction-threshold");

			maxNumLabelsClassifier = configuration.getInt("number-labels-classifier");
			
			weightVotesByFrequency = configuration.getBoolean("weightVotesByFrequency");
			
			betaMemberSelection = configuration.getDouble("beta-member-selection");
					 
			// Set provider settings
			((EagletIndividualCreator) provider).setMaxNumLabelsClassifier(maxNumLabelsClassifier);
			((EagletIndividualCreator) provider).setNumLabels(numberLabels);
						
			// Set evaluator settings
			((MLCEvaluator) evaluator).setTableFitness(tableFitness);
			((MLCEvaluator) evaluator).setTableClassifiers(tableClassifiers);
			((MLCEvaluator) evaluator).setLearner(learner);
			((MLCEvaluator) evaluator).setUseValidationSet(useValidationSet);
			((MLCEvaluator) evaluator).setDatasetTrain(datasetTrain);
			((MLCEvaluator) evaluator).setDatasetValidation(datasetValidation);
			

			// Set genetic operator settings
			((EagletMutator) mutator.getDecorated()).setNumLabels(numberLabels);
			((RandomCrossover) recombinator.getDecorated()).setNumLabels(numberLabels);
			
			//Create randgen
			randgen = randGenFactory.createRandGen();			

		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	

	/**
	 * Initialize algorithm
	 */
	protected void doInit()
	{
		//Calculate individuals by subpopulation
		subpopSize = populationSize / numSubpop;
		
		bset = new ArrayList<List<IIndividual>>(numSubpop);
		pset = new ArrayList<List<IIndividual>>(numSubpop);
		cset = new ArrayList<List<IIndividual>>(numSubpop);
		rset = new ArrayList<List<IIndividual>>(numSubpop);
		
		phiMatrix = new double[numSubpop][numberLabels][numberLabels];
		
		for(int p=0; p<numSubpop; p++) {
			Statistics s = new Statistics();
			
			if((provider.getClass().toString().toLowerCase().contains("phi")) || (mutator.getDecorated().getClass().toString().toLowerCase().contains("phi"))){
				
				try {
					phiMatrix[p] = s.calculatePhi(getDatasetTrain(p));
					
					for(int i=0; i<numberLabels; i++){
						for(int j=0; j<numberLabels; j++)
							if(Double.isNaN(phiMatrix[p][i][j])){
								phiMatrix[p][i][j] = 0;
							}
					}
				} catch (Exception e) {
					e.printStackTrace();
				}
				
				System.out.println("Phi matrix number " + p);
				s.printPhiCorrelations();
				System.out.println();
			}
			
					
			//Pass appearances to provider in case it needs
			if(provider.getClass().toString().toLowerCase().contains("frequency")){
				((FrequencyBasedIndividualCreator) provider).setAppearances(Utils.calculateAppearances(getDatasetTrain(p)));
			}


			if(weightVotesByFrequency){
				//Calculate expected voted based on frequency; ensuring a minimum of 5 votes per label
				weightsPerLabel = Utils.calculateFrequencies(datasetTrain[p]);
				expectedVotesPerLabel = Utils.calculateExpectedVotes(weightsPerLabel, (int)Math.round(3.33*numberLabels)*maxNumLabelsClassifier, numClassifiers, 5, randgen.choose(100));
			}
			else{
				//Spread votes evenly
				weightsPerLabel = new double[numberLabels];
				for(int j=0; j<numberLabels; j++){
					weightsPerLabel[j] = (double)1 / numberLabels;
				}
				expectedVotesPerLabel = Utils.spreadVotesEvenly(numberLabels, (int)Math.round(3.33*numberLabels)*maxNumLabelsClassifier, randgen.choose(100));
			}
			

			System.out.println(Arrays.toString(expectedVotesPerLabel));
			
			/* super.doInit(); */
			// Create individuals
			((EagletIndividualCreator) provider).setSubpopId(p);
			List<IIndividual> prov = provider.provide(subpopSize);
			bset.add(p, prov);
			//System.out.println(bset.get(p).get(0).toString());
			// Evaluate individuals
			((MLCEvaluator)evaluator).evaluate(bset.get(p));
			// Do Control
			doControl();
		}
		
		
	}
	

	protected void doUpdate() 
	{	
		try{
			for(int p=0; p<numSubpop; p++) {
				//Join all bset and cset individuals in cset
				System.out.println("Updating p: " + p);
				//System.out.println("bset.get(p) " + bset.get(p).size());
				//System.out.println("cset.get(p) " + cset.get(p).size());
				cset.get(p).addAll(bset.get(p));
				//System.out.println("cset1.get(p) " + cset.get(p).size());
				cset.set(p, Utils.removeDuplicated(cset.get(p)));
				//System.out.println("cset2.get(p) " + cset.get(p).size());
				
				//List<IIndividual> ensembleMembers = ;
				
				//bset.set(p, bettersSelector.select(cset.get(p), subpopSize));
				bset.set(p, selectEnsembleMembers(cset.get(p), subpopSize, expectedVotesPerLabel, betaMemberSelection));
				
				/*
				List<IIndividual> ensembleMembers = null;
				
				//Select and build ensemble of this generation
				ensembleMembers = selectEnsembleMembers(cset.get(p), numClassifiers, expectedVotesPerLabel, betaMemberSelection);
				//ensembleMembers = randomSelectionFitnessWeighted(cset.get(p), numClassifiers);
				EnsembleMLC currentEnsemble = generateEnsemble(ensembleMembers, numClassifiers, p);
				currentEnsemble.setValidationSet(datasetValidation[p]);
				currentEnsemble.build(datasetTrain[p]);
				
				System.out.println("Ensemble " + p);
				currentEnsemble.printEnsemble();
				System.out.println();
				
				//Evaluate ensemble and compare to the best of all
				EnsembleMLCEvaluator ensembleEval = new EnsembleMLCEvaluator(currentEnsemble, datasetValidation[p]);
					
				iterEnsembleFitness = ensembleEval.evaluate();
				
				System.out.println("Fitness iter " + generation + ": " + iterEnsembleFitness);
				System.out.println("currentEnsemble  votes: " + Arrays.toString(currentEnsemble.getVotesPerLabel()));
				if(iterEnsembleFitness > bestFitness){
					System.out.println("\tNew best fitness!");
					bestFitness = iterEnsembleFitness;
					bestEnsemble = currentEnsemble;
				}
					
				//Add ensemble members to next population
				//Remove ensemble members from cset and select the rest randomly based on fitness
				cset.get(p).removeAll(ensembleMembers);
				bset.set(p, ensembleMembers);
				bset.get(p).addAll(randomSelectionFitnessWeighted(cset.get(p), subpopSize-numClassifiers));
				System.out.println("final bset: " + bset.get(p).size());
				*/
				
				pset.get(p).clear();
				rset.get(p).clear();
				cset.get(p).clear();
				//System.out.println("final bset: " + bset.get(p).size());
			}
		}
		catch(Exception e){
			e.printStackTrace();
		}

		// Clear pset, rset & cset
		//pset.clear();
		//rset.clear();
		//cset.clear();	
	}

	
	@Override
	protected void doControl()
	{
		//System.out.println("------------------------");
		System.out.println("--- Generation " + generation + " ---");
		//System.out.println("------------------------");
		
		
		if(generation % 2 == 0 && generation > 0) {
			
			System.out.println("-- GENERATE ENSEMBLE --");
			
			List<IIndividual> allInds = new ArrayList<IIndividual>();
			for(int p=0; p<numSubpop; p++) {
				allInds.addAll(bset.get(p));
			}
			
			List<IIndividual> ensembleMembers = null;
			try {
				ensembleMembers = selectEnsembleMembers(allInds, numClassifiers, expectedVotesPerLabel, betaMemberSelection);
				EnsembleMLC currentEnsemble = generateEnsemble(ensembleMembers, numClassifiers);
				currentEnsemble.setValidationSet(fullDatasetTrain);
			
				currentEnsemble.build(fullDatasetTrain);
				
				currentEnsemble.printEnsemble();
				
				EnsembleMLCEvaluator ensembleEval = new EnsembleMLCEvaluator(currentEnsemble, fullDatasetTrain);
				iterEnsembleFitness = ensembleEval.evaluate();
				
				System.out.println("Fitness iter " + generation + ": " + iterEnsembleFitness);
				System.out.println("currentEnsemble  votes: " + Arrays.toString(currentEnsemble.getVotesPerLabel()));
				if(iterEnsembleFitness > bestFitness){
					System.out.println("\tNew best fitness!");
					bestFitness = iterEnsembleFitness;
					bestEnsemble = currentEnsemble;
				}
				
				System.out.println();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		
		if(generation >= maxOfGenerations) //generation >= maxOfGenerations
		{
			System.out.println("--- MAX GEN REACHED ---");
			
			ensemble = bestEnsemble;
			System.out.println("Ensemble fitness: " + bestFitness);
			
			System.out.println("Final ensemble");
			System.out.println("--------------");
			ensemble.printEnsemble();
			System.out.println("-----------------");

			System.out.println(Arrays.toString(ensemble.getVotesPerLabel()));
			
			state = FINISHED;
		}
	}
	
	
	private MultiLabelInstances [] generateValidationSet(MultiLabelInstances mlData, ValidationSetTechnique validationSetTechnique){
		/**
		 * 	Array storing train and validation datasets
		 * 		datasets[0] -> train
		 * 		datasets[1] -> validation
		 */
		MultiLabelInstances [] datasets = new MultiLabelInstances[2];
		
		MultiLabelInstances [] folds;
		IterativeStratification strat;

		switch (validationSetTechnique) {
		case pct67:
			strat = new IterativeStratification(seed);
			folds = strat.stratify(mlData, 3);
			
			datasets[0] = folds[0].clone();
			datasets[0].getDataSet().addAll(folds[1].getDataSet());
			datasets[1] = folds[2].clone();			
			break;
			
		case pct75:
			strat = new IterativeStratification(seed);
			folds = strat.stratify(mlData, 4);
			
			datasets[0] = folds[0].clone();
			datasets[0].getDataSet().addAll(folds[1].getDataSet());
			datasets[0].getDataSet().addAll(folds[2].getDataSet());
			datasets[1] = folds[3].clone();
			break;
		
		case pct80:
			strat = new IterativeStratification(seed);
			folds = strat.stratify(mlData, 5);
			
			datasets[0] = folds[0].clone();
			datasets[0].getDataSet().addAll(folds[1].getDataSet());
			datasets[0].getDataSet().addAll(folds[2].getDataSet());
			datasets[0].getDataSet().addAll(folds[3].getDataSet());
			datasets[1] = folds[4].clone();
			break;
		
		case outOfBag:
			ArrayList<Integer> inBag = new ArrayList<Integer>();
			ArrayList<Integer> outOfBag = new ArrayList<Integer>();
			IRandGen randgen = randGenFactory.createRandGen();
			
			for(int i=0; i<mlData.getNumInstances(); i++){
				inBag.add(randgen.choose(0, mlData.getNumInstances()));
			}
			
			//Out of bag includes indices that were not included in the bagged data
			for(int i=0; i<mlData.getNumInstances(); i++){
				if(!inBag.contains(i)){
					outOfBag.add(i);
				}
			}
			
			try {
				datasets[0] = new MultiLabelInstances(mlData.getDataSet(), mlData.getLabelsMetaData());
				Instances data = mlData.getDataSet();
				datasets[0].getDataSet().clear();
				datasets[1] = datasets[0].clone();
				
				for(int element : inBag){
					datasets[0].getDataSet(). add(data.get(element));
				}
				
				for(int element : outOfBag){
					datasets[1].getDataSet(). add(data.get(element));
				}				
			} catch (InvalidDataFormatException e) {
				e.printStackTrace();
			}

			break;
			
		default:
			break;
		}
		
		return datasets;
	}
	
	private EnsembleMLC generateEnsemble(List<IIndividual> members, int n){
		/*byte [][] EnsembleMatrix = new byte[n][numberLabels];
		
		for(int i=0; i<n; i++){
			System.arraycopy(((MultipBinArrayIndividual)members.get(i)).getGenotype(), 0, EnsembleMatrix[i], 0, numberLabels);
		}*/
		
		EnsembleMLC ensemble = new EnsembleMLC(members, learner, numClassifiers, tableClassifiers);
		ensemble.setThreshold(predictionThreshold);
		return ensemble;
		
	}
	
	
	private List<IIndividual> selectEnsembleMembers(List<IIndividual> individuals, int n, int [] expectedVotes, double beta){
		//Copy of the expectedVotes array
		int [] expectedVotesCopy = new int[numberLabels];
		System.arraycopy(expectedVotes, 0, expectedVotesCopy, 0, numberLabels);
		
		//Weights for each label
		double [] weights = weightsPerLabel.clone();
		
//		double beta = 0.5;
		byte [][] EnsembleMatrix = new byte[n][numberLabels];
		
		List<IIndividual> members = new ArrayList<IIndividual>();
		
		List<IIndividual> indsCopy = individuals; // new ArrayList<IIndividual>();
//		indsCopy.addAll(individuals);

		//Sort individuals by fitness
		indsCopy = bettersSelector.select(indsCopy, indsCopy.size());
		
		//Add first individual to ensemble members and remove from list
		System.out.println("Best individual: " + ((MultipBinArrayIndividual)indsCopy.get(0)).toString() + " ; " + ((SimpleValueFitness)indsCopy.get(0).getFitness()).getValue());
		members.add(indsCopy.get(0));
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
				updatedFitnesses[i] = beta * distanceToEnsemble(((MultipBinArrayIndividual)indsCopy.get(i)).getGenotype(), EnsembleMatrix, currentEnsembleSize, weights) + (1-beta)*((SimpleValueFitness)indsCopy.get(i).getFitness()).getValue();
			}
			
			//Get best individual with updated fitness
			int maxIndex = Utils.getMaxIndex(updatedFitnesses);
			
			//Add individual to ensemble members
			members.add(indsCopy.get(maxIndex));
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
			int [] votesPerLabel = Utils.calculateVotesPerLabel(individualsToEnsembleMatrix(members));

			for(int i=0; i<votesPerLabel.length; i++){
				if(votesPerLabel[i] == 0){
					noVotesLabels.add(i);
				}
			}

			if(noVotesLabels.size() > 0){
				weights = Utils.calculateFrequencies(expectedVotes);
				int r = randgen.choose(0, noVotesLabels.size());

				int currentLabel = noVotesLabels.get(r);

				//Remove the worst individual of the most voted label (if do not remove other label appearing only once
				IIndividual worstIndByLabel = getWorstIndividualByLabel(members, Utils.getMaxIndex(votesPerLabel, randgen.choose(100)));
				members.remove(worstIndByLabel);
				
				//Add the individual including label noVotesLabels[r] that better matches with the ensemble
				List<IIndividual> candidates = getIndividualsWithLabel(indsCopy, currentLabel);
				double [] candidatesFitness = new double[candidates.size()];
				
				for(int i=0; i<candidates.size(); i++){
					candidatesFitness[i] = beta * distanceToEnsemble(((MultipBinArrayIndividual)candidates.get(i)).getGenotype(), individualsToEnsembleMatrix(members), members.size(), weights) + (1-beta)*((SimpleValueFitness)candidates.get(i).getFitness()).getValue();
				}
				
				double maxFitness = candidatesFitness[0];
				int maxFitnessIndex = 0;
				for(int i=1; i<candidatesFitness.length; i++){
					if(candidatesFitness[i] > maxFitness){
						maxFitness = candidatesFitness[i];
						maxFitnessIndex = i;
					}
				}
				
				members.add(candidates.get(maxFitnessIndex));
				indsCopy.remove(candidates.get(maxFitnessIndex));
				
				//Re-include the removed indivudual in the indsCopy set.
				indsCopy.add(worstIndByLabel);
			}
		}while(noVotesLabels.size() > 0);
		
		return members;
	}
	
	
	
	private List<IIndividual> randomSelectionFitnessWeighted(List<IIndividual> individuals, int n){
		int nInds = individuals.size();
		double[] fitness = new double[nInds];
		double accFitness = 0;
		
		for(int i=0; i<nInds; i++) {
			fitness[i] = ((SimpleValueFitness)individuals.get(i).getFitness()).getValue();
			accFitness += fitness[i];
		}
		
		List<IIndividual> selected = new ArrayList<IIndividual>();
		double r, acc;
		int last;
		
		//For each individual to select
		for(int i=0; i<n; i++) {
			r = randgen.uniform(0, accFitness);
			//Look for last individual whose acc fitness is lower than r (and not NaN)
			last = -1;
			acc = 0;
			for(int j=0; j<nInds; j++) {
				if(fitness[j] != Double.NaN) {
					acc += fitness[j];
					
					if(r < acc) {
						last = j;
					}
				}
			}
			
			selected.add(individuals.get(last));
			accFitness -= fitness[last];
			fitness[last] = Double.NaN;
		}
		
		return selected;
	}
	
	private IIndividual getWorstIndividualByLabel(List<IIndividual> individuals, int label){
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
	
	private List<IIndividual> getIndividualsWithLabel(List<IIndividual> list, int label){
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
	
	private boolean hasCriticalLabel(IIndividual ind, List<IIndividual> list){
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
	
	
	private double distanceToEnsemble(byte [] ind, byte [][] ensemble, int ensembleSize, double [] weights){
		double distance = 0;
		
		for(int i=0; i<ensembleSize; i++){
			distance += Utils.hammingDistance(ind, ensemble[i], weights);
		}
		
		distance /= ensembleSize;
		
		return distance;
	}
	
	private byte [][] individualsToEnsembleMatrix(List<IIndividual> individuals){
		byte [][] EnsembleMatrix = new byte[individuals.size()][numberLabels];
		
		for(int i=0; i<individuals.size(); i++){
			System.arraycopy(((MultipBinArrayIndividual)individuals.get(i)).getGenotype(), 0, EnsembleMatrix[i], 0, numberLabels);
		}
		
		return EnsembleMatrix;
	}
	
	public double[] updateWeights(double [] weights, double [] evaluationPerLabel){
		double [] updatedW = new double[weights.length];
		
		//Compute the average of evaluationPerLabel
		double avgEvaluation = 0;
		for(double eval : evaluationPerLabel){
			avgEvaluation += eval;
		}
		avgEvaluation /= evaluationPerLabel.length;
		
		//Calculate the ratio of variation of each metric around the mean
		double [] variation = new double[weights.length];
		boolean zeroEvaluation = false;
		for(int i=0; i<weights.length; i++){
			if(evaluationPerLabel[i] < 0.01){
				variation[i] = -1;
				zeroEvaluation = true;
			}
			else{
				variation[i] = avgEvaluation / evaluationPerLabel[i];
			}
		}

		if(zeroEvaluation){
			//Set max variation to labels with 0 evaluation
			double maxVariation = Double.MIN_VALUE;
			for(int i=0; i<variation.length; i++){
				if(variation[i] > maxVariation){
					maxVariation = variation[i];
				}
			}
			for(int i=0; i<variation.length; i++){
				if(Double.compare(variation[i], -1) == 0){
					variation[i] = maxVariation;
				}
			}
		}

		//Update the weights with their variations
		double totalW = 0;
		for(int i=0; i<weights.length; i++){
			updatedW[i] = weights[i] * variation[i];
			totalW += updatedW[i];
		}

		//Make the weights to sum 1
		for(int i=0; i<weights.length; i++){
			updatedW[i] /= totalW;
		}

		return updatedW;
	}
	
	
}
