package eaglet.utils;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import mulan.data.MultiLabelInstances;

/**
 * Class to filter and transform a multi-label dataset given an array of active labels
 * 
 * @author Jose M. Moyano
 *
 */
public class DatasetTransformation {
	
	/**
	 * Original dataset
	 */
	private MultiLabelInstances originalDataset;
	
	/**
	 *  Filtered dataset 
	 */
	private MultiLabelInstances modifiedDataset;
	
	/**
	 *  Array with the indices of the labels to remove
	 */
	private int [] labelsToRemove;
	
	/**
	 *  Binary array that identifies the individual 
	 */
	private byte [] genotype;
	
	/**
	 *  Filter to remove the non-active labels
	 */
	private Remove filter;
	
	
	/**
	 * Constructor
	 * 	
	 * @param dataset Multi-label dataset
	 * @param genotype Genotype of the individual indicating the active labels
	 */
	public DatasetTransformation(MultiLabelInstances dataset, byte [] genotype)
	{
		this.originalDataset = dataset;
		this.genotype = genotype;
		
		filter = new Remove();
		configureFilter();
	}
	

	/**
	 * Sets the genotype with the active labels
	 * 
	 * @param genotype Byte array with active labels
	 */
	public void setGenotype(byte [] genotype)
	{
		this.genotype = genotype;
		
		configureFilter();
	}
	
	/**
	 * Sets the original dataset
	 * 
	 * @param dataset Multi-label dataset to filter
	 */
	public void setOriginalDataset(MultiLabelInstances dataset)
	{
		this.originalDataset = dataset;
	}
	
	/**
	 * Gets the modified dataset
	 * 
	 * @return Modified dataset
	 */
	public MultiLabelInstances getModifiedDataset()
	{
		return modifiedDataset;
	}

	
	/**
	 * Method to transform the dataset
	 */
	public void transformDataset()
	{		
	    
	    try{
	    	//Instances without the selected labels to remove
	    	Instances modified = Filter.useFilter(originalDataset.getDataSet(), filter);
		    
	    	//Create MultiLabelInstances based on the previous Instances
		    modifiedDataset = originalDataset.reintegrateModifiedDataSet(modified);
	    }
	    catch(Exception e)
	    {
	    	e.printStackTrace();
	    }
	}
	
	
	/**
	 * Configures the filter with the indices of the labels to remove
	 */
	private void configureFilter()
	{
		//Obtain label indices
		int [] labelIndices = originalDataset.getLabelIndices();
		
		//Obtain number of active bits in genotype
		int numbLabelsClassifier = 0;
		for(int i=0; i<genotype.length; i++)
		{
			if(genotype[i] == 1)
			{
				numbLabelsClassifier++;
			}
		}
		
		//Array with indices of labels to remove
		labelsToRemove = new int[originalDataset.getNumLabels() - numbLabelsClassifier];
		
		for(int i=0, k=0; k<(originalDataset.getNumLabels() - numbLabelsClassifier); i++)
		{
			if(genotype[i] == 0)
			{
				//Add to labelsToRemove the index of the label (this label is inactive in genotype)
				labelsToRemove[k] = labelIndices[i];
				k++;
			}
		}
		
		//Indicate labels to remove in the dataset
		filter.setAttributeIndicesArray(labelsToRemove);
		
		try{
			filter.setInputFormat(originalDataset.getDataSet());
			filter.setInvertSelection(false);
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
	}

}
