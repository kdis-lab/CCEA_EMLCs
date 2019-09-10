package eaglet.individualCreator;

import java.util.ArrayList;
import java.util.List;

import eaglet.utils.Utils;
import net.sf.jclec.IIndividual;
import net.sf.jclec.binarray.BinArrayIndividual;

/**
 * Class implementing the random generation of individuals
 * 
 * @author Jose M. Moyano
 *
 */
public class RandomIndividualCreator extends EagletIndividualCreator {

	/**
	 * Serialization constant
	 */
	private static final long serialVersionUID = -492984562210436953L;
	
	/**
	 * Constructor
	 */
	public RandomIndividualCreator()
	{
		super();
	}
	
	
	@Override
	public List<IIndividual> provide(int numberOfIndividuals)
	{
		// Set numberOfIndividuals
		this.numberOfIndividuals = numberOfIndividuals;
		// Result list
		createdBuffer = new ArrayList<IIndividual> (numberOfIndividuals);
		// Prepare process
		prepareCreation();
		// Provide individuals
		BinArrayIndividual ind;
		boolean exist;
		for (createdCounter=0; createdCounter<numberOfIndividuals; createdCounter++) {
			do{
				ind = species.createIndividual(createGenotype());
				exist = Utils.exists(ind, createdBuffer);
			}while(exist);

			createdBuffer.add(ind);
		}
		
		
		// Returns result
		return createdBuffer;
	}

	
	/**
	 * Generate random individual genotype
	 * 
	 * @return Genotype
	 */
	public final byte [] createGenotype()
	{
		byte [] genotype = createRandomGenotype();

		return genotype;
	}

}
