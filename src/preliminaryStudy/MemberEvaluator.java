package preliminaryStudy;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Hashtable;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

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
 * @author Jose M. Moyano
 *
 */
public class MemberEvaluator extends AbstractParallelEvaluator {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 7993294994276149645L;

	/**
	 *  Indicates if fitness is to maximize 
	 */
	private boolean maximize = true;
	
	/**
	 *  Fitness comparator 
	 */
	private Comparator<IFitness> COMPARATOR = new ValueFitnessComparator(!maximize);
	
	MultiLabelInstances train;
	MultiLabelInstances validation;
	MultiLabelLearner learner;
	Hashtable<String, MultiLabelLearner> tableClassifiers;

	@Override
	public Comparator<IFitness> getComparator() {
		return COMPARATOR;
	}	

	protected void evaluate(MultipBinArrayIndividual ind) {
		Evaluator eval = new Evaluator();
		
		try {
			//Filter train dataset
			DatasetTransformation dtT = new DatasetTransformation(train, ind.getGenotype());
			dtT.transformDataset();
			MultiLabelInstances newDatasetTrain = dtT.getModifiedDataset();
			
			//System.out.println("train: " + newDatasetTrain.getNumInstances());
			
			MultiLabelLearner mll = learner.makeCopy();
			mll.build(newDatasetTrain);
			
			//System.out.println("learner: " + learner.toString());

			List<Measure> measures = new ArrayList<Measure>();
			measures.add(new ExampleBasedFMeasure());
	       	Evaluation results;
	       	
	       	((LabelPowerset2)mll).setSeed(1);
	       	
	       	if(validation == null) {
	       		results = eval.evaluate(mll, newDatasetTrain, measures);
	       	}
	       	else {
	       		//Filter train validation
	       		//Filter validation dataset
				DatasetTransformation dtV = new DatasetTransformation(validation, ind.getGenotype());
				dtV.transformDataset();
				MultiLabelInstances newDatasetValidation = dtV.getModifiedDataset();
				results = eval.evaluate(mll, newDatasetValidation, measures);
	       	}
	       		       	
	       	double fitness = results.getMeasures().get(0).getValue();
	       	
	       	//System.out.println("Add " + ind.toString() + " to table");
	       	tableClassifiers.put(ind.toString(), mll.makeCopy());
	       	ind.setFitness(new SimpleValueFitness(fitness));
	       	//System.out.println("fitness: " + fitness);
		}
		catch(Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

	@Override
	protected void evaluate(IIndividual ind) {
		System.out.println("ERROR");
		
	}
	
	public void evaluateInds(List<MultipBinArrayIndividual> inds, MultiLabelInstances train, MultiLabelInstances validation,
			MultiLabelLearner learner, Hashtable<String, MultiLabelLearner> tableClassifiers)
	{		
		this.train = train;
		this.validation = validation;
		this.learner = learner;
		this.tableClassifiers = tableClassifiers;
		
		ExecutorService threadExecutor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
		
		for (MultipBinArrayIndividual ind : inds)
		{
			//if (ind.getFitness() == null)
			//{
				threadExecutor.execute(new evaluationThread(ind));
				numberOfEvaluations++;
			//}
		}
		
		threadExecutor.shutdown();
		
		try
		{
			if (!threadExecutor.awaitTermination(30, TimeUnit.DAYS))
				System.out.println("Threadpool timeout occurred");
		}
		catch (InterruptedException ie)
		{
			System.out.println("Threadpool prematurely terminated due to interruption in thread that created pool");
		}

	}

	
	private class evaluationThread extends Thread
	{
		private MultipBinArrayIndividual ind;
		
	    public evaluationThread(MultipBinArrayIndividual ind)
	    {
	        this.ind = ind;
	    }
	    
	    public void run()
	    {
	    	evaluate(ind);
	    }
    }
}
