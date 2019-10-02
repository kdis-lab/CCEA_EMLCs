package eaglet.individualCreator;

import eaglet.utils.Utils;
import net.sf.jclec.binarray.MultipBinArrayCreator;
import net.sf.jclec.binarray.MultipBinArrayIndividual;

/**
 * Class implementing an abstract individual creator class which the rest will extend
 * 
 * @author Jose M. Moyano
 *
 */
public abstract class EagletIndividualCreator extends MultipBinArrayCreator {

	/**
	 * Serialization constant
	 */
	protected static final long serialVersionUID = -5853450005745803105L;

	/**
	 *  Number of labels 
	 */
	protected int numLabels;
	
	/**
	 *  Max number of active labels in the base classifier 
	 */
	protected int maxNumLabelsClassifier;
	
	/**
	 *  Number of min active labels in the base classifier 
	 */
	protected int minNumLabelsClassifier = 2;

	/**
	 *  Indicates if the num of active bits is variable in each individual 
	 */
	protected boolean variable;
	
	
	/**
	 * Constructor
	 */
	public EagletIndividualCreator()
	{
		super();
	}
	
	public EagletIndividualCreator(int p)
	{
		super();
		this.p = p;
	}
		
	/**
	 * Sets the number of labels
	 * 
	 * @param numLabels Number of labels
	 */
	public void setNumLabels(int numLabels)
	{
		this.numLabels = numLabels;
	}
	
	/**
	 * Gets the number of labels
	 * 
	 * @return Number of labels
	 */
	public int getNumLabels()
	{
		return numLabels;
	}
	
	/**
	 * Sets the max number of labels per classifier
	 * 
	 * @param numLabelsClassifier Max number of labels
	 */
	public void setMaxNumLabelsClassifier(int numLabelsClassifier)
	{
		this.maxNumLabelsClassifier = numLabelsClassifier;
	}
	
	/**
	 * Gets the max number of labels per classifier
	 * 
	 * @return Max number of labels
	 */
	public int getMaxNumLabelsClassifier()
	{
		return maxNumLabelsClassifier;
	}
	
	/**
	 * Sets if the number of labels is variable in each classifier
	 * 
	 * @param variable True if the number of labels is variable and false otherwise
	 */
	public void setVariable(boolean variable)
	{
		this.variable = variable;
	}
	
	/**
	 * Replace repeated individuals in createdBuffer by random individuals
	 */
	protected void replaceRepeatedByRandomIndividuals(){
		createdBuffer = Utils.removeDuplicated(createdBuffer);
		
		MultipBinArrayIndividual ind;
		while(createdBuffer.size() < numberOfIndividuals){
			ind = species.createIndividual(createRandomGenotype(), p);
			if(!Utils.exists(ind, createdBuffer)){
				createdBuffer.add(ind);
			}
		}
	}
	
	/**
	 * Generates a random genotype
	 * 
	 * @return Random genotype
	 */
	public byte [] createRandomGenotype()
	{
		byte [] genotype = new byte[numLabels];

		int numLabelsClassifier;
		
		if(variable)
		{
			numLabelsClassifier = randgen.choose(minNumLabelsClassifier, maxNumLabelsClassifier+1);
		}
		else
		{
			numLabelsClassifier = maxNumLabelsClassifier;
		}

        int rand, active = 0;
		do{
            rand = randgen.choose(0, numLabels);
            if(genotype[rand] != 1){
                genotype[rand] = 1;
                active++;
            }
        }while(active < numLabelsClassifier);

		return genotype;
	}
	
	public byte [] createRandomGenotype(int label)
	{
		byte [] genotype = new byte[numLabels];

		int numLabelsClassifier;
		
		if(variable)
		{
			numLabelsClassifier = randgen.choose(minNumLabelsClassifier, maxNumLabelsClassifier+1);
		}
		else
		{
			numLabelsClassifier = maxNumLabelsClassifier;
		}

		genotype[label] = 1;
        int rand, active = 1;
		do{
            rand = randgen.choose(0, numLabels);
            if(genotype[rand] != 1){
                genotype[rand] = 1;
                active++;
            }
        }while(active < numLabelsClassifier);

		return genotype;
	}

}
