package coeaglet.algorithm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Hashtable;
import java.util.List;

import org.apache.commons.configuration.Configuration;

import coeaglet.individualCreator.FrequencyBasedIndividualCreator;
import coeaglet.mutator.Mutator;
import coeaglet.utils.Utils;
import coeaglet.utils.Utils.CommunicationType;
import coeaglet.utils.Utils.EvalType;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.transformation.LabelPowerset2;
import mulan.core.MulanException;
import mulan.data.MultiLabelInstances;
import net.sf.jclec.IIndividual;
import net.sf.jclec.algorithm.classic.MultiSGE;
import net.sf.jclec.fitness.SimpleValueFitness;
import net.sf.jclec.listind.MultipListCreator;
import net.sf.jclec.listind.MultipListIndividual;
import net.sf.jclec.selector.BettersSelector;
import net.sf.jclec.selector.WorsesSelector;
import net.sf.jclec.util.random.IRandGen;
import weka.classifiers.trees.J48;

/**
 * Class implementing the main co-evolutionary algorithm for the optimization of multi-label ensembles. * 
 * It is based on a Multiple populations algorithm (MultiSGE)
 * 
 * CoEAGLETB. More information at:
 * 
 * @author Jose M. Moyano
 *
 */
public class Alg extends MultiSGE {

	/**
	 * Serialization constant
	 */
	private static final long serialVersionUID = 8988750329823184523L;
	
	/**
	 * Number of labels of the dataset
	 */
	int nLabels;
	
	/**
	 *  Number of active labels at each base classifier
	 */
	private int k;
	
	/**
	 * Full training dataset
	 */
	MultiLabelInstances fullTrainData;
	
	/**
	 * Training datasets
	 */
	MultiLabelInstances [] trainData;
	
	/**
	 * Test dataset
	 */
	MultiLabelInstances testData;
	
	/**
	 * Table including the fitness of all individuals
	 */
	Hashtable<String, Double> tableFitness = new Hashtable<String, Double>();
	
	/**
	 * Table including all built classifiers
	 */
	Hashtable<String, MultiLabelLearner> tableClassifiers = new Hashtable<String, MultiLabelLearner>();
	
	/** 
	 * Betters selector.
	 */	
	public BettersSelector bettersSelector = new BettersSelector(this);
	
	/**
	 * Worses selector.
	 */
	private WorsesSelector worseSelector = new WorsesSelector(this);
	
	/**
	 * Generator of random numbers
	 */
	IRandGen randgen;
	
	/**
	 * Best ensemble generated so far
	 */
	Ensemble bestEnsemble;
	
	/**
	 * Fitness of bestEnsemble
	 */
	double bestFitness = -1;
	
	/**
	 * Ratio of instances sampled at each train data
	 */
	double sampleRatio;
	
	/**
	 * Evalaution typa of individuals in the population
	 */
	EvalType evalType;
	
	/**
	 * Type of communication used in the algorithm
	 */
	CommunicationType commType;
	
	/**
	 * Crossover probability in operators communication 
	 */
	double probCrossComm;
	
	/**
	 * Mutation probability in operators communication
	 */
	double probMutComm;
	
	/**
	 * Threshold for ensemble prediction
	 */
	double predictionThreshold;
	
	/**
	 * Number of classifiers in the ensemble
	 */
	int nClassifiers;
	
	/**
	 * Beta value to update subpopulations
	 */
	double betaUpdatePop;
	
	/**
	 * Beta value to build the ensemble
	 */
	double betaEnsembleSelection;
	
	/**
	 * Indicates if the ensemble is pruned or not in the last iteration
	 */
	boolean prune;

	
	/**
	 * Constructor
	 */
	public Alg()
	{
		super();
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
	 * Getter for the ensemble
	 * 
	 * @return Ensemble
	 */
	public Ensemble getEnsemble() {
		return bestEnsemble;
	}
	
	/**
	 * Getter for array of training data
	 * 
	 * @return Array with all sampled training datasets
	 */
	public MultiLabelInstances[] getTrainData() {
		return trainData;
	}
	
	/**
	 * Getter for an specific training data
	 *  
	 * @param p Identifier of subpopulation for the training data 
	 * @return Training data for the corresponding subpopulation
	 */
	public MultiLabelInstances getTrainData(int p) {
		return trainData[p];
	}
	
	/**
	 * Getter for full training dataset
	 * 
	 * @return Full training dataset
	 */
	public MultiLabelInstances getFullTrainData() {
		return fullTrainData;
	}
	
	/**
	 * Getter for test data
	 * @return
	 */
	public MultiLabelInstances getTestData() {
		return testData;
	}
	
	
	
	/**
	 * Configure some default aspects and parameters of EME to make the configuration easier
	 * 
	 * @param configuration Configuration
	 */
	private void configureEagletDefaults(Configuration configuration) {
		//Species
		configuration.setProperty("species[@type]", "net.sf.jclec.listind.MultipListIndividualSpecies");
		configuration.setProperty("species[@genotype-length]", "1");
		
		//Provider (only if not provided)
		if(! configuration.containsKey("provider[@type]")) {
			configuration.addProperty("provider[@type]", "net.sf.jclec.listind.MultipListCreator");
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
			configuration.addProperty("listener[@type]", "net.sf.jclec.listener.PopulationReporter");
		}
		
		//Prediction treshold
		if(! configuration.containsKey("prediction-threshold")) {
			configuration.addProperty("prediction-threshold", "0.5");
		}
	}
	
	
	@Override
	public void configure(Configuration configuration)
	{
		configureEagletDefaults(configuration);
		super.configure(configuration);
		
		randgen = randGenFactory.createRandGen();

		k = configuration.getInt("k");
		predictionThreshold = configuration.getDouble("prediction-threshold");
		nClassifiers = configuration.getInt("number-classifiers");
		betaUpdatePop = configuration.getDouble("beta-update-population");
		betaEnsembleSelection = configuration.getDouble("beta-ensemble-selection");
		prune = configuration.getBoolean("prune-ensemble");

		String datasetTrainFileName = configuration.getString("dataset.train-dataset");
		String datasetTestFileName = configuration.getString("dataset.test-dataset");
		String datasetXMLFileName = configuration.getString("dataset.xml");
		
		sampleRatio = configuration.getDouble("sampling-ratio");
		
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
		}
		
		String evalTypeString = configuration.getString("eval-type");
		switch (evalTypeString) {
		case "train":
			evalType = EvalType.train;
			break;
		case "full":
			evalType = EvalType.full;	
			break;
		}
		
		fullTrainData = null;
		testData = null;
		try {
			fullTrainData = new MultiLabelInstances(datasetTrainFileName, datasetXMLFileName);
			testData = new MultiLabelInstances(datasetTestFileName, datasetXMLFileName);
			
			trainData = new MultiLabelInstances[numSubpop];
			for(int p=0; p<numSubpop; p++) {
				trainData[p] = Utils.sampleData(fullTrainData, sampleRatio, randgen);
			}
		}
		catch(MulanException e) {
			e.printStackTrace();
		}
		
		nLabels = fullTrainData.getNumLabels();
		
		((Mutator) mutator.getDecorated()).setMaxInt(nLabels);
		
		((MultipListCreator) provider).setMaxInt(nLabels);
		((MultipListCreator) provider).setK(k);
		
		((Eval) evaluator).setTrainData(trainData);
		if(evalType == EvalType.full) {
			((Eval) evaluator).setEvalData(fullTrainData);
		}
		else {
			((Eval) evaluator).setEvalData(null);
		}
		((Eval) evaluator).setEvalData(fullTrainData);
		((Eval) evaluator).setTableFitness(tableFitness);
		((Eval) evaluator).setTableClassifiers(tableClassifiers);
	}
	
	/**
	 * Create individuals in population, evaluating before start rest
	 * of evolution
	 */
	protected void doInit() 
	{		
		//Calculate individuals by subpopulation
		subpopSize = (int)Math.round((populationSize*1.0) / numSubpop);
		
		bset = new ArrayList<List<IIndividual>>(numSubpop);
		pset = new ArrayList<List<IIndividual>>(numSubpop);
		cset = new ArrayList<List<IIndividual>>(numSubpop);
		rset = new ArrayList<List<IIndividual>>(numSubpop);
		
		for(int p=0; p<numSubpop; p++) {
			//Initialize each population
			((FrequencyBasedIndividualCreator) provider).setSubpopId(p);
			((FrequencyBasedIndividualCreator) provider).setaMin(3);
			((FrequencyBasedIndividualCreator) provider).setAppearances(Utils.getAppearances(trainData[p]));
			List<IIndividual> prov = provider.provide(subpopSize);
			bset.add(p, prov);
			
			// Evaluate individuals
			//evaluator.evaluate(bset.get(p));
		}
		
		//Evaluate individuals of all subpopulations
		((MultipAbstractParallelEvaluator)evaluator).evaluateMultip(bset);

		// Do Control
		doControl();
	}
	
	
	@Override
	protected void doUpdate() {
		for(int p=0; p<numSubpop; p++) {
			//Add to cset individuals from bset that are not currently included
			for(IIndividual bInd : bset.get(p)) {
				if(!Utils.contains(cset.get(p), (MultipListIndividual) bInd)) {
					cset.get(p).add(bInd);
				}
			}
			
			//Update subpopulation with ensemble selection procedure
			EnsembleSelection eSel = new EnsembleSelection(cset.get(p), bset.get(p).size(), nLabels, betaUpdatePop);
			eSel.setRandgen(randgen);
			eSel.selectEnsemble();
			bset.set(p, eSel.getEnsemble());
			
			IIndividual best = bettersSelector.select(bset.get(p), 1).get(0);
			System.out.println(best + " ; " + ((SimpleValueFitness)best.getFitness()).getValue());
			
			
			//Clear rest of sets
			pset.get(p).clear();
			rset.get(p).clear();
			cset.get(p).clear();
		}
	}
	
	
	/**
	 * At the moment just generate and evaluate the ensemble
	 */
	protected void doCommunication() 
	{
		if((generation % generationsComm == 0) && (generation > 0)) {
			//Join all individuals of all subpopulations
			List<IIndividual> allInds = new ArrayList<IIndividual>();
			for(int p=0; p<numSubpop; p++) {
				allInds.addAll(bset.get(p));
			}
			
			//Add individuals of the best ensemble with a probability
			//	The probability is higher in last generations and lower in earlier
			if(bestEnsemble != null) {
				for(IIndividual ind : bestEnsemble.inds) {
					if(!Utils.contains(allInds, (MultipListIndividual)ind)) {
						if(randgen.coin((generation*1.0)/maxOfGenerations)) {
							allInds.add(ind.copy());
						}
					}
				}
			}
			
			//Create ensemble considering all individuals
			EnsembleSelection eSel = new EnsembleSelection(allInds, nClassifiers, nLabels, betaEnsembleSelection);
			eSel.setRandgen(randgen);
			eSel.selectEnsemble();
			LabelPowerset2 learner = new LabelPowerset2(new J48());
			((LabelPowerset2)learner).setSeed(1);
			
			Ensemble currentEnsemble = new Ensemble(eSel.getEnsemble(), learner);
			currentEnsemble.setTableClassifiers(tableClassifiers);
			try {
				currentEnsemble.build(fullTrainData);
				
				System.out.println(currentEnsemble);
				
				//Evaluate ensemble
				EnsembleEval eEval = new EnsembleEval(currentEnsemble, fullTrainData);
				double eFitness = eEval.evaluate();
				
				System.out.println("Fitness iter " + generation + ": " + eFitness);
				System.out.println("currentEnsemble  votes: " + Arrays.toString(eSel.labelVotes));
				
				if(eFitness > bestFitness) {
					System.out.println("\tNew best fitness!");
					bestEnsemble = currentEnsemble;
					bestFitness = eFitness;
				}
				
				System.out.println();
				
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	
	}
	
	@Override
	protected void doControl()
	{
		System.out.println("Generation " + generation);
		
		if ((generation % generationsComm) == 0 && generation > 0) {
			doCommunication();
		}
		
		if (generation >= maxOfGenerations) {
			if(prune) {
				int nPruned = bestEnsemble.prune(fullTrainData);
				System.out.println(nPruned + " members pruned.");
			}

			EnsembleEval eEval = new EnsembleEval(bestEnsemble, fullTrainData);
			double eFitness = eEval.evaluate();
			System.out.println("FINAL FITNESS: " + eFitness);
			
			state = FINISHED;
			return;
		}
	}	
}
