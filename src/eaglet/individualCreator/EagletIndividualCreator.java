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
	 *  Number of active labels in the base classifier 
	 */
	protected int k;
	
	
	/**
	 * Constructor
	 */
	public EagletIndividualCreator()
	{
		super();
	}
	
	/**
	 * Constructor with subpopulation index
	 * 
	 * @param p Index of subpopulation
	 */
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
	 * Sets the number of active labels per classifier
	 * 
	 * @param k Number of active labels
	 */
	public void setK(int k)
	{
		this.k = k;
	}
	
	/**
	 * Gets the number of active labels per classifier
	 * 
	 * @return Number of active labels at each classifier
	 */
	public int getK()
	{
		return k;
	}
	
	
	/**
	 * Generates a random genotype
	 * 
	 * @return Random genotype
	 */
	public byte [] createRandomGenotype()
	{
		byte [] genotype = new byte[numLabels];

        int rand, active = 0;
		do{
            rand = randgen.choose(0, numLabels);
            if(genotype[rand] != 1){
                genotype[rand] = 1;
                active++;
            }
        }while(active < k);

		return genotype;
	}
	
	/**
	 * Generates an almost random genotype. It will include at least the indicated label.
	 * The rest of active labels are selected randomly.
	 * @param label
	 * @return
	 */
	public byte [] createRandomGenotype(int label)
	{
		byte [] genotype = new byte[numLabels];

		genotype[label] = 1;
        int rand, active = 1;
		do{
            rand = randgen.choose(0, numLabels);
            if(genotype[rand] != 1){
                genotype[rand] = 1;
                active++;
            }
        }while(active < k);

		return genotype;
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

}
