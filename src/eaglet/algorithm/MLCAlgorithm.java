package eaglet.algorithm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
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
import mulan.dimensionalityReduction.BinaryRelevanceAttributeEvaluator;
import mulan.dimensionalityReduction.Ranker;
import net.sf.jclec.IIndividual;
import net.sf.jclec.algorithm.classic.MultiSGE;
import net.sf.jclec.binarray.MultipBinArrayIndividual;
import net.sf.jclec.binarray.MultipBinArraySpecies;
import net.sf.jclec.fitness.SimpleValueFitness;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.transformation.LabelPowerset;
import net.sf.jclec.selector.BettersSelector;
import net.sf.jclec.util.random.IRandGen;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Remove;
import weka.attributeSelection.ChiSquaredAttributeEval;

/**
 * Class implementing the evolutionary algorithm for the optimization of MLCEnsemble
 * It is based on a Multiple populations algorithm (MultiSGE)
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
	 *  Table storing individuals that have been removed or added to a subpopulation
	 *  	in the communication process.
	 *  It avoids from continuously adding the same individuals that previously we removed.
	 */
	private HashSet<String> tabuSet;

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
	 *  If true, individuals are evaluated over different set which they were trained
	 */
	private boolean useValidationSet;
	
	/**
	 * Fitness of the best ensemble at the moment 
	 */
	public double bestFitness = 0.0;
	
	/**
	 * Best ensemble at the moment 
	 */
	private EnsembleMLC bestEnsemble = null;
	
	/**
	 * Fitness of the ensemble in each iteration
	 */
	public double iterEnsembleFitness;
	
	/**
	 * Number of iterations between subpopulations communication
	 */
	public int itersCommunication;
	
	/** 
	 * Betters selector. Used in update phase 
	 */	
	private BettersSelector bettersSelector = new BettersSelector(this);
	
	/**
	 * Techniques for the validation set
	 */
	private enum SamplingTechnique{
		pct67, pct75, pct80, outOfBag, replacement,
	};
	SamplingTechnique samplingTechnique;
	
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
	 * beta value to multiply by distance to the ensemble when updating a subpopulation
	 */
	double betaUpdatePop;
	
	/**
	 * beta value to multiply by distance to the ensemble when selecting members for the ensemble
	 */
	double betaEnsembleSelection;
	
	/**
	 * Indicates how many individuals have been included in the population for the current iteration
	 * at the communication process.
	 */
	int [] currItAdd;
	
	/**
	 * Indicates how many individuals have been removed from the population for the current iteration
	 * at the communication process.
	 */
	int [] currItRem;

	Remove[] filters;
	
	/**
	 * Constructor
	 */
	public MLCAlgorithm()
	{
		super();
		tableFitness = new Hashtable<String, Double>();
		tableClassifiers = new Hashtable<String, MultiLabelLearner>();
		tabuSet = new HashSet<String>();
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
	 * Get the size of the tabu set
	 * 
	 * @return Size of tabu set
	 */
	public int getTabuSize() {
		return tabuSet.size();
	}
	
	/**
	 * Get the number of individuals added to population p in the current iteration of communication
	 * 
	 * @param p Index of subpopulation
	 * @return Current number of individuals added to population p in communication phase
	 */
	public int getCurrItAdd(int p) {
		return currItAdd[p];
	}
	
	/**
	 * Get the number of individuals removed frompopulation p in the current iteration of communication
	 * 
	 * @param p Index of subpopulation
	 * @return Current number of individuals removed from population p in communication phase
	 */
	public int getCurrItRem(int p) {
		return currItRem[p];
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
		currItAdd = new int[numSubpop];
		currItRem = new int[numSubpop];
		filters = new Remove[numSubpop];
		
		try {
			//Get seed for random numbers
			seed = configuration.getLong("rand-gen-factory[@seed]");
			//Create randgen
			randgen = randGenFactory.createRandGen();	
			
			// Read train/test datasets
			String datasetTrainFileName = configuration.getString("dataset.train-dataset");
			String datasetTestFileName = configuration.getString("dataset.test-dataset");
			String datasetXMLFileName = configuration.getString("dataset.xml");
			
			fullDatasetTrain = new MultiLabelInstances(datasetTrainFileName, datasetXMLFileName);
			datasetTest = new MultiLabelInstances(datasetTestFileName, datasetXMLFileName);

			useValidationSet = configuration.getBoolean("validation-set");
			
			String validationSetTechniqueString = configuration.getString("sampling-type");

			switch (validationSetTechniqueString) {
			case "pct67":
				samplingTechnique = SamplingTechnique.pct67;					
				break;
			case "pct75":
				samplingTechnique = SamplingTechnique.pct75;					
				break;
			case "pct80":
				samplingTechnique = SamplingTechnique.pct80;					
				break;
			case "outOfBag":
				samplingTechnique = SamplingTechnique.outOfBag;					
				break;
			case "replacement":
				samplingTechnique = SamplingTechnique.replacement;					
				break;

			default:
				break;
			}
			
			/* If validation set is used:
			 * 	Train set is a subset of full training
			 *  Validation set is full training set (it includes instances not seen in training)
			 */
			if(useValidationSet)
			{
				for(int i=0; i<numSubpop; i++) {
					MultiLabelInstances [] m = generateValidationSet(fullDatasetTrain.clone(), samplingTechnique);
					datasetTrain[i] = m[0];
					datasetValidation[i] = fullDatasetTrain.clone();
					
					//datasetValidation[i] = m[0];
					//datasetValidation[i].getDataSet().addAll(m[1].getDataSet());
					
					//datasetValidation[i] = m[1];
				}
			}
			/* If validation set is not used:
			 * 	Train and validation set are the same; a subset of full training
			 */
			else
			{
				for(int i=0; i<numSubpop; i++) {
					seed = seed + 1;
					MultiLabelInstances [] m = generateValidationSet(fullDatasetTrain.clone(), samplingTechnique);
					
					System.out.println(m[0].getDataSet().numAttributes());

					ASEvaluation ase = new ChiSquaredAttributeEval();
					BinaryRelevanceAttributeEvaluator ae = new BinaryRelevanceAttributeEvaluator(ase, m[0], "avg", "dl", "eval");
					Ranker r = new Ranker();
		            int[] result = r.search(ae, m[0]);
		            
		            int nFeatures = (int)Math.round(m[0].getDataSet().numAttributes() * 0.75);
		            int[] toKeep = new int[nFeatures + m[0].getNumLabels()];
		            System.arraycopy(result, 0, toKeep, 0, nFeatures);
		            int[] labelIndices = m[0].getLabelIndices();
		            System.arraycopy(labelIndices, 0, toKeep, nFeatures, m[0].getNumLabels());
		            
		            filters[i] = new Remove();
		            filters[i].setAttributeIndicesArray(toKeep);
		            filters[i].setInvertSelection(true);
		            filters[i].setInputFormat(m[0].getDataSet());
		            
		            MultiLabelInstances currMlData = new MultiLabelInstances( Filter.useFilter(m[0].getDataSet(), filters[i]), m[0].getLabelsMetaData());
		            
					System.out.println("; " + currMlData.getDataSet().numAttributes());
					
					datasetTrain[i] = currMlData;
					datasetValidation[i] = currMlData;
				}	
			}
			
			//Get number of labels
			numberLabels = fullDatasetTrain.getNumLabels();
			
			numClassifiers = configuration.getInt("number-classifiers");
			predictionThreshold = configuration.getDouble("prediction-threshold");

			maxNumLabelsClassifier = configuration.getInt("number-labels-classifier");
			
			weightVotesByFrequency = configuration.getBoolean("weightVotesByFrequency");
			
			betaUpdatePop = configuration.getDouble("beta-update-population");
			betaEnsembleSelection = configuration.getDouble("beta-ensemble-selection");
					 
			itersCommunication = configuration.getInt("iters-communication");
			
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
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	

	@Override
	protected void doInit()
	{
		//Calculate individuals by subpopulation
		subpopSize = (int)Math.round((populationSize*1.0) / numSubpop);
		
		bset = new ArrayList<List<IIndividual>>(numSubpop);
		pset = new ArrayList<List<IIndividual>>(numSubpop);
		cset = new ArrayList<List<IIndividual>>(numSubpop);
		rset = new ArrayList<List<IIndividual>>(numSubpop);
		
		//Calculate phi matrix only if necessary
		if((provider.getClass().toString().toLowerCase().contains("phi")) || (mutator.getDecorated().getClass().toString().toLowerCase().contains("phi"))){
			phiMatrix = new double[numSubpop][numberLabels][numberLabels];
			Statistics s = new Statistics();
			
			for(int p=0; p<numSubpop; p++) {
				try {
					phiMatrix[p] = s.calculatePhi(getDatasetTrain(p));
					
					for(int i=0; i<numberLabels; i++){
						for(int j=0; j<numberLabels; j++) {
							if(Double.isNaN(phiMatrix[p][i][j])){
								phiMatrix[p][i][j] = 0;
							}
						}	
					}
				} catch (Exception e) {
					e.printStackTrace();
				}
				
				System.out.println("Phi matrix number " + p);
				s.printPhiCorrelations();
			}
		}	
		
		/*
		 * Initialize subpopulations
		 */
		for(int p=0; p<numSubpop; p++) {		
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
			
			// Create individuals
			((EagletIndividualCreator) provider).setSubpopId(p);
			List<IIndividual> prov = provider.provide(subpopSize);
			bset.add(p, prov);

			// Evaluate individuals
			((MLCEvaluator)evaluator).evaluate(bset.get(p));
		}
		
		// Do Control
		doControl();
	}
	
	@Override
	protected void doUpdate() 
	{	
		try{
			for(int p=0; p<numSubpop; p++) {
				//Join all bset and cset individuals in cset; remove duplicated
				cset.get(p).addAll(bset.get(p));
				cset.set(p, Utils.removeDuplicated(cset.get(p)));
				
				//Select individuals from cset, and set as bset
				bset.set(p, selectEnsembleMembers(cset.get(p), subpopSize, expectedVotesPerLabel, betaUpdatePop));
				//bset.set(p, bettersSelector.select(bset.get(p), bset.get(p).size()));
				
				//Clear rest of sets
				pset.get(p).clear();
				rset.get(p).clear();
				cset.get(p).clear();
			}
		}
		catch(Exception e){
			e.printStackTrace();
		}
	}

	
	@Override
	protected void doControl()
	{
		System.out.println("--- Generation " + generation + " ---");		
		
		/* Communicate subpopulations */
		if((generation % itersCommunication == 0) && (generation > 0)) {
			/* Generate ensemble */
			
			//Join all individuals of all subpopulations
			List<IIndividual> allInds = new ArrayList<IIndividual>();
			for(int p=0; p<numSubpop; p++) {
				allInds.addAll(bset.get(p));
			}
			
			System.out.println("antes: " + allInds.size());
			if(bestEnsemble != null) {
				for(IIndividual ind : bestEnsemble.getEnsembleInds()) {
					if(!contains(allInds, ind)) {
						if(randgen.coin((generation*1.0)/maxOfGenerations)) {
							allInds.add(ind);
						}
					}
				}
				//allInds.addAll(bestEnsemble.getEnsembleInds());
				//allInds = Utils.removeDuplicated(allInds);
			}
			System.out.println("despues: " + allInds.size());
			
			//Create an ensemble with all individuals
			EnsembleMLC currentEnsemble = null;
			try {				
				currentEnsemble = generateAndBuildEnsemble(allInds, fullDatasetTrain, numClassifiers, expectedVotesPerLabel, betaEnsembleSelection);
				//currentEnsemble.build(fullDatasetTrain);

				//Evaluate the ensemble
				EnsembleMLCEvaluator ensembleEval = new EnsembleMLCEvaluator(currentEnsemble, fullDatasetTrain);
				iterEnsembleFitness = ensembleEval.evaluate();
				
				System.out.println("Fitness iter " + generation + ": " + iterEnsembleFitness);
				System.out.println("currentEnsemble  votes: " + Arrays.toString(currentEnsemble.getVotesPerLabel()));
				
				//Store ensemble if best ever
				if(iterEnsembleFitness > bestFitness){
					System.out.println("\tNew best fitness!");
					bestFitness = iterEnsembleFitness;
					bestEnsemble = currentEnsemble;
					currentEnsemble.printEnsemble();
				}

				System.out.println();
				
				//Communicate subpopulations (it will modified the bsets)
				bset = communicateSubpops(bset, currentEnsemble.getEnsembleInds());
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		/* Algorithm finished; Get best ensemble */
		if(generation >= maxOfGenerations)
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
	private EnsembleMLC generateAndBuildEnsemble(List<IIndividual> individuals, MultiLabelInstances mlData, int n, int [] expectedVotes, double beta) {
		List<IIndividual> ensembleMembers = null;
		EnsembleMLC ensemble = null;
		
		try {
			ensembleMembers = selectEnsembleMembers(individuals, n, expectedVotes, beta);
			ensemble = generateEnsemble(ensembleMembers);
		
			ensemble.build(mlData);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return ensemble;
	}
	
	/**
	 * Do communication between subpopulations
	 * 
	 * @param bset List of bsets of each population
	 * @param ensemble List of individuals included in the ensemble created with all previous individuals
	 * @return New list of bsets for each subpopulation after communication
	 */
	public List<List<IIndividual>> communicateSubpops(List<List<IIndividual>> bset, List<IIndividual> ensemble) {
		HashSet<String> set = new HashSet<String>();
		String comb = null;
		
		//Reset the count of individuals added/removed from each subpopulation
		for(int p=0; p<numSubpop; p++) {
			currItAdd[p] = 0;
			currItRem[p] = 0;
		}
		
		for(int i=0; i<ensemble.size(); i++) {
			//For each individual in the ensemble, get its combination of labels
			comb = Arrays.toString(((MultipBinArrayIndividual)ensemble.get(i)).getGenotype()); 
			
			//The same combination could appear more than once in the ensemble
				//We will consider only first one
			if(! set.contains(comb)) {
				set.add(comb);
				
				//For each subpopulation, except the one of the current individual,
					//Update subpopulation based on the given individual
				for(int p=0; p<numSubpop; p++) {
					if(p != ((MultipBinArrayIndividual)ensemble.get(i)).getSubpop()) {
						updateSubpop(bset.get(p), ensemble.get(i));
					}
				}
				
				if(!containsComb(bset.get(((MultipBinArrayIndividual)ensemble.get(i)).getSubpop()), (MultipBinArrayIndividual)ensemble.get(i))) {
					if(randgen.coin((generation*1.0)/maxOfGenerations)) {
						bset.get(((MultipBinArrayIndividual)ensemble.get(i)).getSubpop()).add(ensemble.get(i));
						System.out.println("Add it!");
					}
				}
			}
		}
		
		for(int p=0; p<numSubpop; p++) {
			//Evaluate new added individuals
			((MLCEvaluator)evaluator).evaluate(bset.get(p));
			
			//If more individuals than allowed, select individuals
			if(bset.get(p).size() > subpopSize) {
				bset.set(p, selectEnsembleMembers(bset.get(p), subpopSize, expectedVotesPerLabel, betaUpdatePop));
			}
			//If less individuals than needed, add and evaluate random individuals
			else if(bset.get(p).size() < subpopSize) {
				fillPopRandom(bset.get(p), subpopSize);
				((MLCEvaluator)evaluator).evaluate(bset.get(p));
			}
		}
		
		return bset;
	}
	
	/**
	 * Fill population (or subpopulation) with random individuals
	 * 
	 * @param pop List of individuals defininf a (sub)population
	 * @param toReach Number of individuals to reach in the population
	 */
	protected void fillPopRandom(List<IIndividual> pop, int toReach){
		MultipBinArrayIndividual ind;
		int p = ((MultipBinArrayIndividual)pop.get(0)).getSubpop();
		
		while(pop.size() < toReach) {
			ind = ((MultipBinArraySpecies)species).createIndividual(((EagletIndividualCreator) provider).createRandomGenotype(), p);
			//pop.contains(ind);
			if(!containsComb(pop, ind)) {
				pop.add(ind);
			}
		}
	}
	
	/**
	 * Check if a list of individuals contains the combination of labels of a given individual
	 * 
	 * @param list List of individuals
	 * @param ind Single individual
	 * @return True if it contains and false otherwise
	 */
	public boolean containsComb(List<IIndividual> list, IIndividual ind) {
		for(IIndividual oInd : list) {
			if(Arrays.toString(((MultipBinArrayIndividual)ind).getGenotype()).equals(Arrays.toString(((MultipBinArrayIndividual)oInd).getGenotype()))) {
				return true;
			}
		}
		return false;
	}
	
	public boolean contains(List<IIndividual> list, IIndividual ind) {
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
	 * Update subpopulation given an individual appearing in the ensemble
	 * 
	 * @param list List of individuals of the subpopulation
	 * @param ind Individual included in the ensemble
	 * @return Returns 0 if not updated, 1 if an individual was added, and -1 if an individual was removed
	 */
	public int updateSubpop(List<IIndividual> list, IIndividual ind) {
		//If the individual is included in the tabu set (it has been previously evaluated for update)
		//It has only 25% probabilities of being again included on the population
		if(tabuSet.contains(((MultipBinArrayIndividual)ind).toString())){
			if(randgen.coin(0.75)) {
			//if(randgen.coin( 1 - ((generation*1.0)/maxOfGenerations)) ) {
				//With probability, subpopulation will not be updated with same individual
				return 0;
			}
		}
		
		for(IIndividual oInd : list) {
			if(Arrays.toString(((MultipBinArrayIndividual)ind).getGenotype()).equals(Arrays.toString(((MultipBinArrayIndividual)oInd).getGenotype()))) {
				//If the combination of labels in the individual is present in the ensemble
					//remove useless individual of population p
					//(It is better predicted by individual ind than by oInd).				
				list.remove(oInd);
				tabuSet.add(((MultipBinArrayIndividual)ind).toString());
				currItRem[((MultipBinArrayIndividual)ind).getSubpop()]++;
				return -1;
			}
		}
		
		//If the combination was not present in the population, maybe it would be better predicted with this data
		//Add it with a probability
			//0.5 prob if it is not in tabu set
			//0.125 prob if it is in tabu set
		double coinProb = 0.5;
		//coinProb = 1 - ((generation*1.0) / maxOfGenerations);
		/*if(tabuSet.contains(((MultipBinArrayIndividual)ind).toString())){
			coinProb = 0.125;
		}*/
		
		if(randgen.coin(coinProb)) {
			MultipBinArrayIndividual newInd = new MultipBinArrayIndividual(((MultipBinArrayIndividual)ind).getGenotype(), ((MultipBinArrayIndividual)list.get(0)).getSubpop());
			list.add(newInd);
			tabuSet.add(((MultipBinArrayIndividual)newInd).toString());
			currItAdd[((MultipBinArrayIndividual)ind).getSubpop()]++;
			return +1;
		}
		
		return 0;
	}
	
	/**
	 * Partition data into train and validation sets
	 * 
	 * @param mlData Full data
	 * @param samplingTechnique Technique for selecting the data
	 * @return
	 */
	private MultiLabelInstances [] generateValidationSet(MultiLabelInstances mlData, SamplingTechnique samplingTechnique){
		/**
		 * 	Array storing train and validation datasets
		 * 		datasets[0] -> train
		 * 		datasets[1] -> validation
		 */
		MultiLabelInstances [] datasets = new MultiLabelInstances[2];
		
		MultiLabelInstances [] folds;
		IterativeStratification strat;
		
		Instances data, newData;

		switch (samplingTechnique) {
		case replacement:
			data = mlData.getDataSet();
			newData = new Instances(data);
			newData.removeAll(newData);
			
			for(int i=0; i<data.numInstances(); i++) {
				newData.add(data.get(randgen.choose(data.numInstances())));
			}
			
			try {
				datasets[0] = new MultiLabelInstances(newData, mlData.getLabelsMetaData());
			} catch (InvalidDataFormatException e1) {
				e1.printStackTrace();
			}
			datasets[1] = null;
			
			break;
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
				data = mlData.getDataSet();
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
	
	/**
	 * Generate an ensemble given the list of individuals
	 * 
	 * @param members Members of the ensemble
	 * @return Ensemble generated
	 */
	private EnsembleMLC generateEnsemble(List<IIndividual> members){
		EnsembleMLC ensemble = new EnsembleMLC(members, learner, numClassifiers, tableClassifiers);
		ensemble.setThreshold(predictionThreshold);
		ensemble.setFilters(filters);
		return ensemble;
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
	private List<IIndividual> selectEnsembleMembers(List<IIndividual> individuals, int n, int [] expectedVotes, double beta){
		//Copy of the expectedVotes array
		int [] expectedVotesCopy = new int[numberLabels];
		System.arraycopy(expectedVotes, 0, expectedVotesCopy, 0, numberLabels);
		
		//Weights for each label
		double [] weights = weightsPerLabel.clone();
		
		byte [][] EnsembleMatrix = new byte[n][numberLabels];
		
		List<IIndividual> members = new ArrayList<IIndividual>();
		
		List<IIndividual> indsCopy = individuals; // new ArrayList<IIndividual>();
//		indsCopy.addAll(individuals);

		//Sort individuals by fitness
		indsCopy = bettersSelector.select(indsCopy, indsCopy.size());
		
		//Add first individual to ensemble members and remove from list
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
				updatedFitnesses[i] = beta * distanceToEnsemble(indsCopy.get(i), members, currentEnsembleSize, weights, 0.75) + (1-beta)*((SimpleValueFitness)indsCopy.get(i).getFitness()).getValue();
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
					candidatesFitness[i] = beta * distanceToEnsemble(candidates.get(i), members, members.size(), weights, 0.75) + (1-beta)*((SimpleValueFitness)candidates.get(i).getFitness()).getValue();
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
	
	/**
	 * Get the individual with less predictive performance from those containing a given label
	 * 
	 * @param individuals List of individuals
	 * @param label Given label
	 * @return Worst individual with the label
	 */
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
	
	/**
	 * Obtain all individuals containing a given label
	 * 
	 * @param list List of individuals
	 * @param label Label
	 * @return
	 */
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
	
	/**
	 * Indicates if an individual has a critical label.
	 * It is a critical label if it only appears once in the population
	 * 
	 * @param ind Individual to check
	 * @param list List of individuals
	 * @return True if any of the labels in ind is critical
	 */
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
	
	/**
	 * Calculate the distance from one individual to an ensemble
	 * 
	 * @param ind Individual
	 * @param ensemble Ensemble
	 * @param ensembleSize Size of the ensemble
	 * @param weights Weights for each label to calculate la distance
	 * @return Distance from individual to ensemble
	 */
	private double distanceToEnsemble(byte [] ind, byte [][] ensemble, int ensembleSize, double [] weights){
		double distance = 0;
		
		for(int i=0; i<ensembleSize; i++){
			distance += Utils.hammingDistance(ind, ensemble[i], weights);
		}
		
		distance /= ensembleSize;
		
		return distance;
	}
	
	private double distanceToEnsemble(IIndividual ind, List<IIndividual> ensemble, int ensembleSize, double [] weights, double ratioTrain){
		double distance = 0;
		
		for(int i=0; i<ensembleSize; i++){
			distance += Utils.distance((MultipBinArrayIndividual)ind, (MultipBinArrayIndividual)ensemble.get(i), weights, ratioTrain);
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
	private byte [][] individualsToEnsembleMatrix(List<IIndividual> individuals){
		byte [][] EnsembleMatrix = new byte[individuals.size()][numberLabels];
		
		for(int i=0; i<individuals.size(); i++){
			System.arraycopy(((MultipBinArrayIndividual)individuals.get(i)).getGenotype(), 0, EnsembleMatrix[i], 0, numberLabels);
		}
		
		return EnsembleMatrix;
	}
	
}
