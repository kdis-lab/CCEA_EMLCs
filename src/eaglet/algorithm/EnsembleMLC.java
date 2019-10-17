package eaglet.algorithm;

import java.util.Hashtable;
import java.util.List;

import eaglet.utils.Utils;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import mulan.classifier.InvalidDataException;
import mulan.classifier.ModelInitializationException;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.meta.MultiLabelMetaLearner;
import mulan.classifier.transformation.LabelPowerset2;
import mulan.data.MultiLabelInstances;
import net.sf.jclec.IIndividual;
import net.sf.jclec.binarray.MultipBinArrayIndividual;

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
	 * List of individuals identifying the ensemble
	 */
	private List<IIndividual> EnsembleInds;
	
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
	 * Seed for ties in prediction
	 */
	int seed;
	
	/**
	 * All training data
	 */
	MultiLabelInstances [] trainData;
	
	
	/**
	 * Constructor 
	 * 
	 * @param EnsembleInds IIndividuals identifying the base classifiers of the ensemble
	 * @param baseLearner Multi-label learner to use in the ensemble
	 * @param numClassifiers Number of classifiers
	 * @param tableClassifiers Table storing the previously built classifiers
	 */
	public EnsembleMLC(List<IIndividual> EnsembleInds, MultiLabelLearner baseLearner, int numClassifiers, Hashtable<String, MultiLabelLearner> tableClassifiers, MultiLabelInstances [] trainData)
	{
		super(baseLearner);
		
		this.EnsembleInds = EnsembleInds;		
		this.numClassifiers = numClassifiers;
		this.tableClassifiers = tableClassifiers;
		this.trainData = trainData;
	}
	
	/**
	 * Copy contstructor
	 * 
	 * @param e Ensemble to copy
	 */
	public EnsembleMLC(EnsembleMLC e)
	{
		super(e.baseLearner);
		
		this.EnsembleInds = e.EnsembleInds;		
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
	 * Set the seed to resolve ties in prediction
	 * 
	 * @param seed Seed
	 */
	public void setSeed(int seed) {
		this.seed = seed;
	}
	
	/**
	 * Get the number of classsifiers of the ensemble
	 * 
	 * @return Number of classifiers
	 */
	public int getNumClassifiers()
	{
		return EnsembleInds.size();
	}
	
	/**
	 * Sets the number of classifiers of the ensemble
	 * 
	 * @param n Number of classifiers
	 */
	public void setNumClassifiers(int n) {
		this.numClassifiers = n;
	}
	
	/**
	 * Gets the ensemble matrix
	 * 
	 * @return Ensemble matrix
	 */
	public List<IIndividual> getEnsembleInds()
	{
		return EnsembleInds;
	}
	
	/**
	 * Gets the ensemble matrix (byte matrix identifying the ensemble).
	 * It is calculated given the individuals of the ensemble.
	 * 
	 * @return Byte ensemble matrix
	 */
	public byte[][] getEnsembleMatrix(){
		byte [][] mat = new byte[numClassifiers][numLabels];
		
		for(int i=0; i<numClassifiers;i++) {
			mat[i] = ((MultipBinArrayIndividual)EnsembleInds.get(i)).getGenotype();
		}
		
		return mat;
	}
		
	/**
	 * Calculate the number of votes per label in the ensemble
	 * 
	 * @return Array with the number of votes per label
	 */
	public int[] calculateVotesPerLabel()
	{	
		votesPerLabel = new int[numLabels];
		
		byte [][] EnsembleMatrix = getEnsembleMatrix();
		
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

	
	@Override
	public String toString()
	{
		String str = "";
		str+="\nnumLabels: "+numLabels;
		str+="\nnumClassifiers:"+numClassifiers;
		str+="\nEnsembleMatrix:\n";		
		
		
		for(int model=0; model<numClassifiers; model++)
		{	
			str += ((MultipBinArrayIndividual)EnsembleInds.get(model)).toString();
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
				
				if(tableClassifiers == null) {
					Ensemble[i] = Utils.buildLearner((MultipBinArrayIndividual) EnsembleInds.get(i), trainData[((MultipBinArrayIndividual) EnsembleInds.get(i)).getSubpop()]);
				}
				else {
					String s = EnsembleInds.get(i).toString();
					Ensemble[i] = tableClassifiers.get(s);
				}
				
				
			}

			//Sets wieghts evenly
			for(int i = 0; i < numClassifiers; i++)
			{
				for(int j=0; j<numLabels; j++){
					voteWeights[i][j] = (double)1 / votesPerLabel[j];
				}
			}
	}
	
	@Override
	protected MultiLabelOutput makePredictionInternal(Instance instance)
			throws Exception, InvalidDataException 	
	{						
	    double[] sumConf = new double[numLabels];
	    double[] sumVotes = new double[numLabels];
	    
	    byte[][] EnsembleMatrix = getEnsembleMatrix();
	    
	    for(int model = 0; model < numClassifiers; model++) {
	    	((LabelPowerset2)Ensemble[model]).setSeed(seed);
	    }
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

	     MultiLabelOutput mlo = new MultiLabelOutput(bipartition, sumVotes);
	    
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
		System.out.println(this.toString());
	}
	
	/**
	 * Obtain confidences of the ensemble for a given instance
	 * 
	 * @param instance Instance of the dataset
	 * @return Array with confidences for the given instance
	 */
	protected double [] getConfidences(Instance instance)
			throws Exception, InvalidDataException 	
	{				
	    double[] sumConf = new double[numLabels];
	    
	    byte[][] EnsembleMatrix = getEnsembleMatrix();
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
	
	
	
}
