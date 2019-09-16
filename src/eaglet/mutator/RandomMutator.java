package eaglet.mutator;

import net.sf.jclec.binarray.MultipBinArrayIndividual;

/**
 * Class implementing the bits exchange mutator
 * 
 * @author Jose M. Moyano
 *
 */
public class RandomMutator extends EagletMutator {

	/**
	 * Serialization constant
	 */
	private static final long serialVersionUID = 2455293830055566959L;
	
	/**
	 * Constructor
	 */
	public RandomMutator()
	{
		super();
	}
	
	@Override
	public boolean equals(Object other)
	{
		if (other instanceof RandomMutator) {
			return true;
		}
		else {
			return false;
		}
	}
	
	
	@Override
	protected void mutateNext() {
		//Get individual to be mutated
		MultipBinArrayIndividual mutant = (MultipBinArrayIndividual) parentsBuffer.get(parentsCounter);
		
		int gl = mutant.getGenotype().length;
		
		// Creates mutant genotype
		byte [] mgenome = new byte[gl];
		
		System.arraycopy(mutant.getGenotype(), 0, mgenome, 0, gl);

		//Select two bits with different values
		int mp1 = randgen.choose(0, gl);
		int mp2;
		do{
			mp2 = randgen.choose(0, gl);
		}while(mgenome[mp1]==mgenome[mp2]);
			
		// Swap
		byte aux = mgenome[mp1];
		mgenome[mp1] = mgenome[mp2];
		mgenome[mp2] = aux;
		
		sonsBuffer.add(species.createIndividual(mgenome, mutant.getSubpop()));
	}
	
}
