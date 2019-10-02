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
	 * Number of subpops
	 */
	int nSubpops;
	
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
	
	/**
	 * Sets the number of subpopulations
	 * 
	 * @param n Number of subpopulations
	 */
	public void setNumSubpopulations(int n) {
		this.nSubpops = n;
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
	
	/**
	 * Mutate the subpopulation of a given individual.
	 * It changes its subpopulation index for other different index
	 * 
	 * @param mutant Individual to mutate
	 * @return Mutated individual
	 */
	public MultipBinArrayIndividual mutateIndSubpop(MultipBinArrayIndividual mutant) {
		int mSubpop = mutant.getSubpop();
		int newSubpop;

		do{
			newSubpop = randgen.choose(0, nSubpops);
		}while(newSubpop != mSubpop);
		
		mutant.setSubpop(newSubpop);
		
		return mutant;
	}
	
}
