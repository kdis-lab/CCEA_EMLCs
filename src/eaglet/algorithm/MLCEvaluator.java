package eaglet.algorithm;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Hashtable;
import java.util.List;

import eaglet.utils.DatasetTransformation;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.transformation.LabelPowerset2;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.measure.ExampleBasedFMeasure;
import mulan.evaluation.measure.Measure;
import net.sf.jclec.IFitness;
import net.sf.jclec.IIndividual;
import net.sf.jclec.base.AbstractParallelEvaluator;
import net.sf.jclec.binarray.MultipBinArrayIndividual;
import net.sf.jclec.fitness.SimpleValueFitness;
import net.sf.jclec.fitness.ValueFitnessComparator;

/**
 * Class implementing the evaluator of the individuals (MLC base classifiers)
 * 
 * @author Jose M. Moyano
 *
 */
public class MLCEvaluator extends AbstractParallelEvaluator {

	/**
	 * Serialization constant
	 */
	private static final long serialVersionUID = 3488129084072488337L;

	
	/**
	 *  Dataset to build the base classifiers 
	 */
	private MultiLabelInstances[] datasetTrain;
	
	/**
	 *  Dataset to evaluate the individuals 
	 */
	private MultiLabelInstances[] datasetValidation;
	
	/**
	 *  Indicates if fitness is to maximize 
	 */
	private boolean maximize = true;
	
	/**
	 *  Fitness comparator 
	 */
	private Comparator<IFitness> COMPARATOR = new ValueFitnessComparator(!maximize);
	
	/**
	 *  Table that stores the fitness of all individuals that have been evaluated 
	 */
	private Hashtable<String, Double> tableFitness;
	
	/**
	 *  Table that stores all the base classifiers that have been built 
	 */
	private Hashtable<String, MultiLabelLearner> tableClassifiers;
	
	/** 
	 * Multi-label base classifier 
	 */
	private MultiLabelLearner learner;
	
	/**
	 *  Indicates if a validation set is used to evaluate the individuals 
	 */
	private boolean useValidationSet;
	
	/**
	 * Seed to resolve ties in prediction
	 */
	private int seed;
	
	
	/**
	 * Constructor
	 */
	public MLCEvaluator()
	{
		super();
	}
	
	
	/**
	 * Sets the train dataset
	 * 
	 * @param datasetTrain Multi-label train dataset
	 */
	public void setDatasetTrain(MultiLabelInstances[] datasetTrain)
	{
		this.datasetTrain = datasetTrain;
	}
	
	
	/**
	 * Gets if a validation dataset is used to evaluate the individuals
	 * 
	 * @return true if a validation set is used and false otherwise
	 */
	public boolean getUseValidationSet()
	{
		return useValidationSet;
	}
	
	/**
	 * Sets the validation dataset
	 * 
	 * @param datasetValidation Multi-label validation dataset
	 */
	public void setDatasetValidation(MultiLabelInstances[] datasetValidation)
	{
		this.datasetValidation = datasetValidation;
	}
	
	/**
	 * Sets the table with the fitness of the previously evaluated individuals
	 * 
	 * @param tableFitness Hashtable that stores the fitness values
	 */
	public void setTableFitness(Hashtable<String, Double> tableFitness)
	{
		this.tableFitness = tableFitness;
	}
	
	/**
	 * Sets the table with the previously built base classifiers
	 * 
	 * @param tableClassifiers Hashtable that stores the classifiers
	 */
	public void setTableClassifiers(Hashtable<String, MultiLabelLearner> tableClassifiers)
	{
		this.tableClassifiers = tableClassifiers;
	}
	
	/**
	 * Sets the multi-label classifier to use
	 * 
	 * @param learner Multi-label learner
	 */
	public void setLearner(MultiLabelLearner learner)
	{
		this.learner = learner;
	}
	
	/**
	 * Set the seed to resolve ties in prediction
	 * 
	 * @param seed Seed
	 */
	public void setSeed(int seed) {
		this.seed = seed;
	}

	/**
	 * Sets if a validation set is used to evaluate the individuals
	 * 
	 * @param isValidationSet True if a validation set is used and false otherwise
	 */
	public void setUseValidationSet(boolean isValidationSet)
	{
		this.useValidationSet = isValidationSet;
	}

	@Override
	public Comparator<IFitness> getComparator() {
		return COMPARATOR;
	}	
	
	@Override
	protected void evaluate(IIndividual ind) 
	{
		// Individual genotype
		byte[] genotype = ((MultipBinArrayIndividual) ind).getGenotype();
		
		int p = ((MultipBinArrayIndividual) ind).getSubpop();

		//Genotype to string
		String s = ((MultipBinArrayIndividual) ind).toString();
		
		double fitness = -1;
		
		if(tableFitness.containsKey(s))
		{
			//If it has been evaluated yet, it is obtained from the table
			fitness = tableFitness.get(s);
		}
		else
		{
			Evaluator eval = new Evaluator();
			
			try{
				//Filter train dataset
				DatasetTransformation dtT = new DatasetTransformation(datasetTrain[p], genotype);
				dtT.transformDataset();
				MultiLabelInstances newDatasetTrain = dtT.getModifiedDataset();
				
				//Build multilabel learner
				MultiLabelLearner mll = learner.makeCopy();
				mll.build(newDatasetTrain);
				

				MultiLabelInstances newDatasetValidation = null;
				if(useValidationSet)
				{
					//Filter validation dataset
					DatasetTransformation dtV = new DatasetTransformation(datasetValidation[p], genotype);
					dtV.transformDataset();
					newDatasetValidation = dtV.getModifiedDataset();
				}
				else
				{
					newDatasetValidation = newDatasetTrain;
				}
				
				//Evaluate
				List<Measure> measures = new ArrayList<Measure>();
				measures.add(new ExampleBasedFMeasure());

		       	Evaluation results;		       
		       	
		       	((LabelPowerset2)mll).setSeed(seed);
		       	results = eval.evaluate(mll, newDatasetValidation, measures);
		       	//System.out.println("results: " + results);
		       	//((LabelPowerset2)mll).setSeed(1);
		       	//results = eval.evaluate(mll, newDatasetValidation, measures);
		       	//System.out.println("results2: " + results);
		       	
	     	  	fitness = results.getMeasures().get(0).getValue();
	     	  	
	     	  	//Put fitness and built classifier in tables
	     	  	tableFitness.put(s, fitness);
	     	  	tableClassifiers.put(s, mll.makeCopy());
				
			} catch (IllegalArgumentException e) {
				e.printStackTrace();
			} catch (Exception e) {
				e.printStackTrace();
			}	
		}
		
		//Set individual fitness
		ind.setFitness(new SimpleValueFitness(fitness));
	}

}
