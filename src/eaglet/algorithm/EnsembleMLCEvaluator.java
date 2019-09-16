package eaglet.algorithm;

import java.util.ArrayList;
import java.util.List;

import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.measure.ExampleBasedFMeasure;
import mulan.evaluation.measure.Measure;

/**
 * Class implementing the evaluation of an EnsembleMLC
 * 
 * @author Jose M. Moyano
 *
 */
public class EnsembleMLCEvaluator {
		
	/**
	 *  Ensemble classifier to evaluate 
	 */
	private EnsembleMLC ensemble;
	
	/**
	 *  Dataset to evaluate the ensemble
	 */
	private MultiLabelInstances mlData;

	/**
	 *  Fitness value of the ensemble 
	 */
	private double fitness;
	
	
	/**
	 * Constructor
	 * 
	 * @param ensemble Ensemble
	 * @param dataset Multi-label dataset
	 */
	public EnsembleMLCEvaluator(EnsembleMLC ensemble, MultiLabelInstances dataset)
	{
		this.ensemble = ensemble;
		this.mlData = dataset;
		this.fitness = Double.MIN_VALUE;
	}
	
	/**
	 * Sets the dataset to evaluate
	 * 
	 * @param dataset Multi-label dataset
	 */
	public void setDataset(MultiLabelInstances dataset)
	{
		this.mlData = dataset;
	}
	
	/**
	 * Gets the fitness value of the ensemble
	 * 
	 * @return Fitness value
	 */
	public double getFitness()
	{
		return fitness;
	}
	
	/**
	 * Evaluates the ensemble
	 * 
	 * @return Fitness value of the ensemble
	 */
	public double evaluate()
	{		
		List<Measure> measures = new ArrayList<Measure>();  	       
     	measures.add(new ExampleBasedFMeasure());
     	
     	Evaluation results;     	
     	Evaluator eval = new Evaluator();
     	
     	try {
			results = eval.evaluate(ensemble, mlData, measures);
			fitness = results.getMeasures().get(0).getValue();
		} catch (Exception e) {
			e.printStackTrace();
		}
     			
		return fitness;
	}


}
