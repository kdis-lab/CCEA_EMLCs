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
import eaglet.mutator.RandomMutator;
import eaglet.recombinator.RandomCrossover;
import eaglet.utils.Utils;
import eaglet.utils.Utils.CommunicationType;
import eaglet.utils.Utils.EvalType;
import eaglet.utils.Utils.SamplingTechnique;
import mulan.data.InvalidDataFormatException;
import mulan.data.IterativeStratification;
import mulan.data.MultiLabelInstances;
import mulan.data.Statistics;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.measure.ExampleBasedFMeasure;
import mulan.evaluation.measure.Measure;
import net.sf.jclec.IIndividual;
import net.sf.jclec.algorithm.classic.MultiSGE;
import net.sf.jclec.binarray.MultipBinArrayIndividual;
import net.sf.jclec.binarray.MultipBinArraySpecies;
import net.sf.jclec.fitness.SimpleValueFitness;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.transformation.LabelPowerset2;
import net.sf.jclec.selector.BettersSelector;
import net.sf.jclec.util.random.IRandGen;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomTree;
import weka.core.Instances;

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
	 *  Datasets to build the base classifiers 
	 */
	private MultiLabelInstances[] datasetsTrain;
	
	/**
	 *  Dataset to evaluate the final ensemble 
	 */
	private MultiLabelInstances datasetTest;
	
	/**
	 * Number of labels of the dataset
	 */
	int numberLabels;
	
	/**
	 *  Number of active labels at each base classifier
	 */
	private int k;
	
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
	 *  	in the exchange communication process.
	 *  It avoids from continuously adding the same individuals that previously we removed.
	 */
	private HashSet<String> tabuSet;

	/**
	 *  Ensemble classifier 
	 */
	private EnsembleMLC ensemble;
	
	/**
	 * Evalaution typa of individuals in the population
	 */
	Utils.EvalType evalType;
	
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
	 * Sampling data technique used
	 */
	Utils.SamplingTechnique samplingTechnique;
	
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
	 * Indicates if more frequent labels have more chance to have more votes
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
	 * at the exchange communication process.
	 */
	int [] currItAdd;
	
	/**
	 * Indicates how many individuals have been removed from the population for the current iteration
	 * at the exchange communication process.
	 */
	int [] currItRem;
	
	/**
	 * Type of communication used in the algorithm
	 */
	Utils.CommunicationType commType;
	
	/**
	 * Probability of crossover individuals in operators communication
	 */
	double probCrossComm;
	
	/**
	 * Probability of mutate individuals in operators communication
	 */
	double probMutComm;
	
	/**
	 * Indicates if the final ensemble is pruned at the end
	 */
	boolean prune;
	
	/**
	 * Average expected votes per label
	 */
	int avgVotes = 10;
	
	
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
		learner = null;
	}
	
	
	/**
	 * Gets the train multi-label dataset
	 * 
	 * @return Multi-label train dataset
	 */
	public MultiLabelInstances getDatasetTrain(int i)
	{
		return datasetsTrain[i];
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
	public int getk()
	{
		return k;
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
		datasetsTrain = new MultiLabelInstances[numSubpop];
		currItAdd = new int[numSubpop];
		currItRem = new int[numSubpop];
		
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
			
			String sampleTypeString = configuration.getString("sampling-type");

			switch (sampleTypeString) {
			case "pct67":
				samplingTechnique = SamplingTechnique.pct67;					
				break;
			case "pct75":
				samplingTechnique = SamplingTechnique.pct75;					
				break;
			case "pct80":
				samplingTechnique = SamplingTechnique.pct80;					
				break;

			default:
				break;
			}
			
			String commTypeString = configuration.getString("communication");

			switch (commTypeString) {
			case "no":
				commType = CommunicationType.no;					
				break;
			case "exchange":
				commType = CommunicationType.exchange;					
				break;
			case "operators":
				commType = CommunicationType.operators;
				probCrossComm = configuration.getDouble("probability-crossover-communication");
				probMutComm = configuration.getDouble("probability-mutator-communication");
				break;
			
			default:
				break;
			}
			
			String evalTypeString = configuration.getString("eval-type");

			switch (evalTypeString) {
			case "train":
				evalType = EvalType.train;
				break;
			case "full":
				evalType = EvalType.full;	
				break;
				
			default:
				break;
			}
			
			for(int i=0; i<numSubpop; i++) {
				seed = seed + 1;
				datasetsTrain[i] = Utils.sampleData(fullDatasetTrain.clone(), samplingTechnique, randgen);
			}
			
			// Set base learner
			//RandomTree rt = new RandomTree();
			//rt.setKValue((int)Math.round(fullDatasetTrain.getDataSet().numAttributes() * .75));
			J48 j48 = new J48();
			learner = new LabelPowerset2(j48);
			((LabelPowerset2)learner).setSeed((int)seed);
			
			//Get number of labels
			numberLabels = fullDatasetTrain.getNumLabels();
			
			numClassifiers = configuration.getInt("number-classifiers");
			predictionThreshold = configuration.getDouble("prediction-threshold");

			k = configuration.getInt("number-labels-classifier");
			
			weightVotesByFrequency = configuration.getBoolean("weightVotesByFrequency");
			
			betaUpdatePop = configuration.getDouble("beta-update-population");
			betaEnsembleSelection = configuration.getDouble("beta-ensemble-selection");
					 
			itersCommunication = configuration.getInt("iters-communication");
			
			prune = configuration.getBoolean("prune-ensemble");
			
			// Set provider settings
			((EagletIndividualCreator) provider).setK(k);
			((EagletIndividualCreator) provider).setNumLabels(numberLabels);
						
			// Set evaluator settings
			((MLCEvaluator) evaluator).setTableFitness(tableFitness);
			((MLCEvaluator) evaluator).setTableClassifiers(tableClassifiers);
			((MLCEvaluator) evaluator).setLearner(learner);
			((MLCEvaluator) evaluator).setDatasetsTrain(datasetsTrain);
			if(evalType == EvalType.full) {
				((MLCEvaluator) evaluator).setDatasetValidation(fullDatasetTrain);
			}
			((MLCEvaluator) evaluator).setSeed((int)seed);
			
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
		
		//Initialice space for each subpopulation
		bset = new ArrayList<List<IIndividual>>(numSubpop);
		pset = new ArrayList<List<IIndividual>>(numSubpop);
		cset = new ArrayList<List<IIndividual>>(numSubpop);
		rset = new ArrayList<List<IIndividual>>(numSubpop);
		
		/*
		 * Initialize subpopulations
		 */
		for(int p=0; p<numSubpop; p++) {		
			//Pass appearances to provider in case it needs
			if(provider.getClass().toString().toLowerCase().contains("frequency")){
				((FrequencyBasedIndividualCreator) provider).setAppearances(Utils.calculateAppearances(getDatasetTrain(p)));
			}

			//Spread votes evenly
			weightsPerLabel = new double[numberLabels];
			for(int j=0; j<numberLabels; j++){
				weightsPerLabel[j] = (double)1 / numberLabels;
			}
			expectedVotesPerLabel = Utils.spreadVotesEvenly(numberLabels, (int)Math.round(((avgVotes*1.0)/k)*numberLabels)*k, randgen.choose(100));
			

			System.out.println(Arrays.toString(expectedVotesPerLabel));
			
			// Create individuals
			((EagletIndividualCreator) provider).setSubpopId(p);
			List<IIndividual> prov = provider.provide(subpopSize);
			bset.add(p, prov);
		}
		
		//Evaluate all individuals
		((MLCEvaluator)evaluator).evaluateMultip(bset);
		
		// Do Control
		doControl();
	}
	
	@Override
	protected void doUpdate() 
	{	
		try{
			//Update each subpop independently
			for(int p=0; p<numSubpop; p++) {
				//Join all bset and cset individuals in cset; remove duplicated
				cset.get(p).addAll(bset.get(p));
				cset.set(p, Utils.removeDuplicated(cset.get(p)));
				
				//If one subpopulation remains less individuals than required, include random inds
				if(cset.get(p).size() < subpopSize) {
					MultipBinArrayIndividual ind;
					
					while(cset.get(p).size() < subpopSize){
						ind = ((MultipBinArraySpecies)species).createIndividual(((EagletIndividualCreator) provider).createRandomGenotype(), p);
						if(!Utils.exists(ind, cset.get(p))){
							cset.get(p).add(ind);
						}
					}
					
					//Evaluate new random individuals
					((MLCEvaluator)evaluator).evaluateMultip(cset);
				}
				
				//Select individuals from cset, and set as bset
				bset.set(p, selectEnsembleMembers(cset.get(p), subpopSize, expectedVotesPerLabel, betaUpdatePop));
				
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
		if((commType != CommunicationType.no) && (generation % itersCommunication == 0) && (generation > 0)) {
			/* Generate ensemble */
				
			//Join all individuals of all subpopulations
			List<IIndividual> allInds = new ArrayList<IIndividual>();
			for(int p=0; p<numSubpop; p++) {
				allInds.addAll(bset.get(p));
			}
				
			//Add individuals of the best ensemble with a probability
			//	The probability is higher in last generations and lower in earlier
			if(bestEnsemble != null) {
				for(IIndividual ind : bestEnsemble.getEnsembleInds()) {
					if(!Utils.contains(allInds, ind)) {
						if(randgen.coin((generation*1.0)/maxOfGenerations)) {
							allInds.add(ind);
						}
					}
				}
			}
				
			//Create an ensemble considering all individuals
			EnsembleMLC currentEnsemble = null;
			try {
				currentEnsemble = generateAndBuildEnsemble(allInds, fullDatasetTrain, numClassifiers, expectedVotesPerLabel, betaEnsembleSelection, false);

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
					
				//Communicate subpopulations (it will modify the bsets)
				bset = communicateSubpops(bset, currentEnsemble.getEnsembleInds(), commType);
			} catch (Exception e) {
				e.printStackTrace();
			}			
		}
		
		/* Algorithm finished; Get best ensemble */
		if(generation >= maxOfGenerations)
		{
			System.out.println("--- MAX GEN REACHED ---");
			
			if(bestEnsemble == null) {
				//Join all individuals
				List<IIndividual> allInds = new ArrayList<IIndividual>();
				for(int p=0; p<numSubpop; p++) {
					allInds.addAll(bset.get(p));
				}
				
				//Generate ensemble with all individuals
				ensemble = generateAndBuildEnsemble(allInds, fullDatasetTrain, numClassifiers, expectedVotesPerLabel, betaEnsembleSelection, true);
				
				//Evaluate final ensemble
				EnsembleMLCEvaluator ensembleEval = new EnsembleMLCEvaluator(ensemble, fullDatasetTrain);
				bestFitness = ensembleEval.evaluate();
			}
			else {
				//Call to the generateAndBuildEnsemble method with the individuals of the best ensemble AND prune = true
				//It just try to prune the best ensemble
				ensemble = generateAndBuildEnsemble(bestEnsemble.getEnsembleInds(), fullDatasetTrain, numClassifiers, expectedVotesPerLabel, betaEnsembleSelection, true);
			}
			

			System.out.println("Ensemble fitness: " + bestFitness);
			
			System.out.println("Final ensemble: " + ensemble.getNumClassifiers() + " classifiers.");
			System.out.println("--------------");
			ensemble.printEnsemble();
			System.out.println("-----------------");

			System.out.println(Arrays.toString(ensemble.getVotesPerLabel()));
			
			state = FINISHED;
		}
	}
	
	
	/***
	 * OTHER METHODS
	 */	
	
	
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
	private EnsembleMLC generateAndBuildEnsemble(List<IIndividual> individuals, MultiLabelInstances mlData, int n, int [] expectedVotes, double beta, boolean prune) {
		List<IIndividual> ensembleMembers = null;
		EnsembleMLC ensemble = null;
		
		try {
			ensembleMembers = selectEnsembleMembers(individuals, n, expectedVotes, beta);
			if(prune) {
				ensembleMembers = pruneEnsemble(ensembleMembers, mlData, learner, tableClassifiers);
			}
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
	public List<List<IIndividual>> communicateSubpops(List<List<IIndividual>> bset, List<IIndividual> ensemble, CommunicationType commType) {
		if(commType == CommunicationType.exchange) {
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
					
					if(!Utils.containsComb(bset.get(((MultipBinArrayIndividual)ensemble.get(i)).getSubpop()), (MultipBinArrayIndividual)ensemble.get(i))) {
						if(randgen.coin((generation*1.0)/maxOfGenerations)) {
							bset.get(((MultipBinArrayIndividual)ensemble.get(i)).getSubpop()).add(ensemble.get(i));
							//System.out.println("Add it!");
						}
					}
				}
			}
			
			//Check if we removed all individuals for any of the labels
			for(int p=0; p<numSubpop; p++) {
				int[] votes = Utils.calculateVotesPerLabel(bset.get(p), numberLabels);
				for(int i=0; i<votes.length; i++) {
					if(votes[i] <= 0) {
						bset.get(p).add(((MultipBinArraySpecies)species).createIndividual(((EagletIndividualCreator) provider).createRandomGenotype(i), p));
					}
				}
			}
			
			((MLCEvaluator)evaluator).evaluateMultip(bset);
			for(int p=0; p<numSubpop; p++) {
				//Evaluate new added individuals
				//((MLCEvaluator)evaluator).evaluate(bset.get(p));
				
				//If more individuals than allowed, select individuals
				if(bset.get(p).size() > subpopSize) {					
					bset.set(p, selectEnsembleMembers(bset.get(p), subpopSize, expectedVotesPerLabel, betaUpdatePop));
				}
				//If less individuals than needed, add and evaluate random individuals
				else if(bset.get(p).size() < subpopSize) {
					bset.set(p, Utils.fillPopRandom(bset.get(p), subpopSize, species, provider));
					((MLCEvaluator)evaluator).evaluate(bset.get(p));
				}
			}
		}
		else if(commType == CommunicationType.operators) {
			IIndividual randInd;
			MultipBinArrayIndividual[] crossedInds;
			List<MultipBinArrayIndividual> newInds = new ArrayList<MultipBinArrayIndividual>();

			((RandomMutator) mutator.getDecorated()).setNumSubpopulations(numSubpop);
			
			// For each subpopulation, cross and mutate
			for(int p=0; p<numSubpop; p++) {
				//System.out.println("\tsize1: " + bset.get(p).size());
				//For each individual
				for(IIndividual ind : bset.get(p)) {
					//Do crossover with random ind from other subpop
					if(randgen.coin(probCrossComm)) {
						randInd = Utils.selectRandomIndividual(bset, p, randgen);
						crossedInds = ((RandomCrossover) recombinator.getDecorated()).recombineInds((MultipBinArrayIndividual) ind, (MultipBinArrayIndividual) randInd);
						newInds.add(crossedInds[0]);
						newInds.add(crossedInds[1]);
					}
					
					if(randgen.coin(probMutComm)) {
						newInds.add(((RandomMutator) mutator.getDecorated()).mutateIndSubpop((MultipBinArrayIndividual)ind));
					}
				}
			}
			
			int subpop;
			//Move each new individual to its corresponding subpop
			for(IIndividual ind : newInds) {
				subpop = ((MultipBinArrayIndividual)ind).getSubpop();
				bset.get(subpop).add(ind);
			}
			
			//Evaluate new individuals
			((MLCEvaluator)evaluator).evaluateMultip(bset);
			
			//Select members for each subpop
			for(int p=0; p<numSubpop; p++) {
				if(bset.get(p).size() > subpopSize) {
					//System.out.println("\tsize2: " + bset.get(p).size());
					bset.set(p, selectEnsembleMembers(bset.get(p), subpopSize, expectedVotesPerLabel, betaUpdatePop));
				}
			}
		}
		
		return bset;
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
	 * Generate an ensemble given the list of individuals
	 * 
	 * @param members Members of the ensemble
	 * @return Ensemble generated
	 */
	private EnsembleMLC generateEnsemble(List<IIndividual> members){
		return generateEnsemble(members, learner, members.size(), tableClassifiers);
	}
	
	/**
	 * Generate an ensemble given the list of individuals to use and some other parameters
	 * @param members List of members for the ensemble
	 * @param learner Multi-label base classifier to use in each member
	 * @param numClassifiers Number of classifiers in the ensemble
	 * @param tableClassifiers Table including each classifier previously built
	 * @return Ensemble generated
	 */
	private EnsembleMLC generateEnsemble(List<IIndividual> members, MultiLabelLearner learner, int numClassifiers, Hashtable<String, MultiLabelLearner> tableClassifiers){
		EnsembleMLC ensemble = new EnsembleMLC(members, learner, numClassifiers, tableClassifiers);
		ensemble.setThreshold(predictionThreshold);
		ensemble.setSeed((int)seed);
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
		
		List<IIndividual> indsCopy = new ArrayList<IIndividual>();
		indsCopy.addAll(individuals);

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
				updatedFitnesses[i] = beta * Utils.distanceToEnsemble(indsCopy.get(i), members, currentEnsembleSize, weights) + (1-beta)*((SimpleValueFitness)indsCopy.get(i).getFitness()).getValue();
			}
			
			//System.out.println("\t\t" + Arrays.toString(updatedFitnesses));
			//Get best individual with updated fitness
			int maxIndex = Utils.getMaxIndex(updatedFitnesses, randgen.choose(100));
			
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
			int [] votesPerLabel = Utils.calculateVotesPerLabel(Utils.individualsToEnsembleMatrix(members));

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
				IIndividual worstIndByLabel = Utils.getWorstIndividualByLabel(members, Utils.getMaxIndex(votesPerLabel, randgen.choose(100)));
				members.remove(worstIndByLabel);
				
				//Add the individual including label noVotesLabels[r] that better matches with the ensemble
				List<IIndividual> candidates = Utils.getIndividualsWithLabel(indsCopy, currentLabel);
				double [] candidatesFitness = new double[candidates.size()];
				
				for(int i=0; i<candidates.size(); i++){
					candidatesFitness[i] = beta * Utils.distanceToEnsemble(candidates.get(i), members, members.size(), weights) + (1-beta)*((SimpleValueFitness)candidates.get(i).getFitness()).getValue();
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
	 * Given a list of members of an ensemble, it tries to prune the ensemble in reverse order of individuals.
	 * It considers the predictive performance of the ensemble to prune it.
	 * It it decreases the performance by removing more than 10% of members, it stop trying to prune.
	 * 
	 * @param members Members of the ensemble
	 * @param mlData Data to validate the prediction accuracy of the ensemble
	 * @param learner Multi-label classifier to use in each member
	 * @param tableClassifiers Table including each classifier built so far
	 * @return
	 */
	public List<IIndividual> pruneEnsemble(List<IIndividual> members, MultiLabelInstances mlData,
			MultiLabelLearner learner, Hashtable<String, MultiLabelLearner> tableClassifiers) {
		EnsembleMLC ensemble = null;
		List<IIndividual> bestMembers = new ArrayList<IIndividual>(members);
		List<IIndividual> copyMembers = new ArrayList<IIndividual>(members);
		
		double bestFit = 0.0, currFit;
		int nWorst = (int)Math.round(members.size() * .1);
		int failed = 0;
		
		try {
			ensemble = generateEnsemble(copyMembers, learner, copyMembers.size(), tableClassifiers);
			ensemble.build(mlData);
			
			Evaluation results;     	
	     	Evaluator eval = new Evaluator();
	     	List<Measure> measures = new ArrayList<Measure>();  	
	     	measures.add(new ExampleBasedFMeasure());
	     	results = eval.evaluate(ensemble, mlData, measures);
	     	bestFit = results.getMeasures().get(0).getValue();
	     	System.out.println("Fitness: " + bestFit);
	     	
	     	while(failed < nWorst) {
	     		copyMembers.remove(copyMembers.size()-1);
	     		ensemble = generateEnsemble(copyMembers, learner, copyMembers.size(), tableClassifiers);
				ensemble.build(mlData);
		     	results = eval.evaluate(ensemble, mlData, measures);
		     	currFit = results.getMeasures().get(0).getValue();
	     		
		     	if(currFit > bestFit) {
		     		bestFit = currFit;
		     		bestMembers = new ArrayList<IIndividual>(ensemble.getEnsembleInds());
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
	
	
	
}
