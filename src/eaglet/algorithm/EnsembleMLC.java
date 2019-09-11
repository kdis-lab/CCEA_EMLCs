package eaglet.algorithm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Hashtable;

import eaglet.utils.Utils;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import mulan.classifier.InvalidDataException;
import mulan.classifier.ModelInitializationException;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.meta.MultiLabelMetaLearner;
import mulan.data.MultiLabelInstances;
import mulan.data.Statistics;

/**
 * Class implementing the MLC Ensemble
 * 
 * @author Jose M. Moyano
 *
 */
public class EnsembleMLC extends MultiLabelMetaLearner {

	/**
	 * Serialization constant	
	 */
	private static final long serialVersionUID = -7522102456194018593L;

	
	/**
	 *  Number of classifier in the ensemble 
	 */
	private int numClassifiers;
	
	/**
	 *  Threshold for voting process 
	 */
	private double threshold;
	
	/**
	 *  Array with all base classifiers of the ensemble 
	 */
	private MultiLabelLearner [] Ensemble;
	
	/**
	 *  Binary matrix identifying the ensemble 
	 */
	private byte [][] EnsembleMatrix;
	
	/**
	 *  Table that stores all base classifiers that have been built 
	 */
	private Hashtable<String, MultiLabelLearner> tableClassifiers;
	
	/**
	 *  Array with the number of votes of the ensemble for each label 
	 */
	private int [] votesPerLabel;
		
	/**
	 * Weights for the vote of the ensemble
	 */
	double [][] voteWeights;
	
	/**
	 * Validation dataset to calculate weights
	 */
	MultiLabelInstances validationSet;
	
	/**
	 * Identifier of subpopulation
	 */
	int p;
	
	/**
	 * Constructor 
	 * 
	 * @param EnsembleMatrix Matrix identifying the base classifiers of the ensemble
	 * @param baseLearner Multi-label learner to use in the ensemble
	 * @param numClassifiers Number of classifiers
	 * @param threshold Threshold for voting
	 * @param tableClassifiers Table storing the previously built classifiers
	 * @param tablePerformanceByLabel Table storing the performances per label
	 */
	public EnsembleMLC(byte[][] EnsembleMatrix, MultiLabelLearner baseLearner, int numClassifiers, Hashtable<String, MultiLabelLearner> tableClassifiers, int p)
	{
		super(baseLearner);
		
		this.EnsembleMatrix = EnsembleMatrix;		
		this.numClassifiers = numClassifiers;
		this.tableClassifiers = tableClassifiers;
		this.p = p;
	}
	
	public EnsembleMLC(EnsembleMLC e)
	{
		super(e.baseLearner);
		
		this.EnsembleMatrix = e.EnsembleMatrix;		
		this.numClassifiers = e.numClassifiers;
		this.tableClassifiers = e.tableClassifiers;
	}
	
	/**
	 * Gets the threshold value
	 * 
	 * @return threshold
	 */
	public double getThreshold()
	{
		return threshold;
	}
	
	/**
	 * Sets the threshold value
	 * 
	 * @param threshold Threshold value
	 */
	public void setThreshold(double threshold){
		this.threshold = threshold;
	}
	
	/**
	 * Get the number of classsifiers of the ensemble
	 * 
	 * @return Number of classifiers
	 */
	public int getnumClassifiers()
	{
		return numClassifiers;
	}
	
	/**
	 * Gets the ensemble matrix
	 * 
	 * @return Ensemble matrix
	 */
	public byte[][] getEnsembleMatrix()
	{
		return EnsembleMatrix;
	}
		
	/**
	 * Calculate the number of votes per label in the ensemble
	 * 
	 * @return Array with the number of votes per label
	 */
	public int[] calculateVotesPerLabel()
	{	
		votesPerLabel = new int[numLabels];
		
		for(int i=0; i<EnsembleMatrix.length; i++)
		{
			for(int j=0; j<EnsembleMatrix[0].length; j++)
			{
				votesPerLabel[j] += EnsembleMatrix[i][j];
			}
		}
		
		return votesPerLabel;
	}
	
	/**
	 * Gets an array with the number of votes per label
	 * 
	 * @return Array with the number of votes per label 
	 */
	public int[] getVotesPerLabel() {
		return votesPerLabel;
	}
	
	/**
	 * Sets the validation set to calculate weights
	 * 
	 * @param validationSet Multi-label validation dataset
	 */
	public void setValidationSet(MultiLabelInstances validationSet){
		this.validationSet = validationSet;
	}
	
	
	@Override
	public String toString()
	{
		String str = "";
		str+="\nnumLabels: "+numLabels;
		str+="\nnumClassifiers:"+numClassifiers;
		str+="\nEnsembleMatrix:\n";		
		for(int model=0; model<numClassifiers; model++)
		{	
			for (int label=0; label<numLabels; label++)
			{	
				if(EnsembleMatrix[model][label]==0)
				    str+="0 ";
				else
					str+="1 ";
			}
			str+="\n";
		}
		return str;
	}

	/**
	 * Classify a multi-label dataset
	 * 
	 * @param mlData Multi-label dataset
	 * @return Matrix with the label predictions for all instances
	 */
	public int[][] classify(MultiLabelInstances mlData)
	{
		int[][] predictions = new int[mlData.getNumInstances()][numLabels];

		Instances data = mlData.getDataSet();
		
		for (int i=0; i<mlData.getNumInstances(); i++)
		{ 	
		    try {
				MultiLabelOutput mlo = this.makePrediction(data.get(i));
				for(int j=0; j<this.numLabels; j++)
				{	
				  if(mlo.getBipartition()[j])
				  {
					  predictions[i][j]=1;
				  }	
				  else
				  {
					  predictions[i][j]=0;
				  }	  
				}
				
			} catch (InvalidDataException e) {
				e.printStackTrace();
			} catch (ModelInitializationException e) {
				e.printStackTrace();
			} catch (Exception e) {
				e.printStackTrace();
			}
		    
		}
		return(predictions);		
	}	
	
	
	@Override
	protected void buildInternal(MultiLabelInstances trainingData)
			throws Exception {		
			//All base classifiers were built before
				//Now, all of them are in tableClassifiers
		
			//Ensemble of multi-label classifiers
		   	Ensemble = new MultiLabelLearner[numClassifiers];
		   	
		   	//Weights for each base classifier
		   	voteWeights = new double[numClassifiers][numLabels];
		   
			calculateVotesPerLabel();
			
			for(int i = 0; i < numClassifiers; i++)
			{
				//Get classifier from table
				String s = p + Arrays.toString(EnsembleMatrix[i]);
				System.out.println("---" + s);
				Ensemble[i] = tableClassifiers.get(s);
				System.out.println("\t\t" + Ensemble[i]);
				if(Ensemble[i] == null) {
					System.out.println("String " + s + " not found in the table.");
					System.exit(-1);
				}
			}

			//Sets wieghts evenly
			for(int i = 0; i < numClassifiers; i++)
			{
				for(int j=0; j<numLabels; j++){
					voteWeights[i][j] = (double)1 / votesPerLabel[j];
				}
			}
			
			System.out.println("Ensemble built! " + Ensemble.length);
			//System.out.println(Ensemble[0].toString());
	}
	
	@Override
	protected MultiLabelOutput makePredictionInternal(Instance instance)
			throws Exception, InvalidDataException 	
	{				
		//System.out.println("ENSEMBLE");
		//System.out.println(Ensemble.toString());;
		
	    double[] sumConf = new double[numLabels];
	    double[] sumVotes = new double[numLabels];
	    
	    //Gather votes
		for(int model = 0; model < numClassifiers; model++) 
	    {
			//System.out.println(Ensemble[model].toString());
			if(Ensemble == null) {
				System.out.println("Ensemble is null");
				System.exit(-1);
			}
			//System.out.println("Ensemble length: " + Ensemble.length);
			//System.out.println("\t"+Ensemble[0].toString());
			if(Ensemble[model] == null) {
				System.out.println("Ensemble[model] is null");
				System.exit(-1);
			}
			if(instance == null) {
				System.out.println("instance is null");
				System.exit(-1);
			}
			MultiLabelOutput subsetMLO = Ensemble[model].makePrediction(instance);
			        	
			for(int label=0, k=0; label < numLabels; label++)
			{  
				if(EnsembleMatrix[model][label]==1)
				{	
					//Calculate the product between the confidence/bipartition and the corresponding weight
					sumConf[label] += subsetMLO.getConfidences()[k] * voteWeights[model][label];
					sumVotes[label] += subsetMLO.getBipartition()[k] ? voteWeights[model][label] : 0;
					k++;
				}
			}
	     }
			  
	     boolean[] bipartition = new boolean[numLabels];
			        
	     for(int i = 0; i < numLabels; i++)
	     {			         
	    	 if(sumVotes[i] >= threshold){
        		 bipartition[i] = true;
        	 }
        	 else{
        		 bipartition[i] = false;
        	 }
	     }

	     MultiLabelOutput mlo = null;
	     
	     mlo = new MultiLabelOutput(bipartition, sumVotes);
	    
	     return mlo;
	}	 
	
	
	@Override
	public TechnicalInformation getTechnicalInformation() {
		return null;
	}

	@Override
	public String globalInfo() {
		String str = "";
		str+="Class implementing the final ensemble";
		return str;
	}
	
	/**
	 * Prints an ensemble by stdout
	 */
	public void printEnsemble()
	{
		for(int i=0; i<EnsembleMatrix.length; i++)
		{
			System.out.println(Arrays.toString(EnsembleMatrix[i]));
		}
		System.out.println();
		System.out.println("Ensemble size: " + EnsembleMatrix.length + " base classifiers");
	}
	

	protected double [] getConfidences(Instance instance)
			throws Exception, InvalidDataException 	
	{				
	    double[] sumConf = new double[numLabels];
	    
	    //Gather votes
		for(int model = 0; model < numClassifiers; model++) 
	    {
			MultiLabelOutput subsetMLO = Ensemble[model].makePrediction(instance);
			        	
			for(int label=0, k=0; label < numLabels; label++)
			{  
				if(EnsembleMatrix[model][label]==1)
				{	
					//Calculate the product between the confidence/bipartition and the corresponding weight
					sumConf[label] += subsetMLO.getConfidences()[k] * voteWeights[model][label];
					k++;
				}
			}
	     }

	     return sumConf;
	}	 
	
	
	protected MultiLabelOutput makePredictionInternal(Instance instance, double threshold)
			throws Exception, InvalidDataException 	
	{				
	    double[] sumConf = new double[numLabels];
	    double[] sumVotes = new double[numLabels];
	    
	    //Gather votes
		for(int model = 0; model < numClassifiers; model++) 
	    {
			MultiLabelOutput subsetMLO = Ensemble[model].makePrediction(instance);
			        	
			for(int label=0, k=0; label < numLabels; label++)
			{  
				if(EnsembleMatrix[model][label]==1)
				{	
					//Calculate the product between the confidence/bipartition and the corresponding weight
					sumConf[label] += subsetMLO.getConfidences()[k] * voteWeights[model][label];
					sumVotes[label] += subsetMLO.getBipartition()[k] ? voteWeights[model][label] : 0;
					k++;
				}
			}
	     }
			  
	     boolean[] bipartition = new boolean[numLabels];
			        
	     for(int i = 0; i < numLabels; i++)
	     {			         
	    	 if(sumVotes[i] >= threshold){
        		 bipartition[i] = true;
        	 }
        	 else{
        		 bipartition[i] = false;
        	 }
	     }

	     MultiLabelOutput mlo = null;
	     
	     mlo = new MultiLabelOutput(bipartition, sumVotes);
	    
	     return mlo;
	}
}
