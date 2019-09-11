package net.sf.jclec.binarray;

import net.sf.jclec.ISpecies;

import net.sf.jclec.base.AbstractRecombinator;

/**
 * BinArrayIndividual (and subclasses) specific recombinator.  
 * 
 * @author Sebastian Ventura
 */

public abstract class MultipBinArrayRecombinator extends BinArrayRecombinator
{
	
	/////////////////////////////////////////////////////////////////
	// ------------------------------------------ Operation variables
	/////////////////////////////////////////////////////////////////

	/** Individual species (taked from execution context) */
	
	protected transient MultipBinArraySpecies species;

	/////////////////////////////////////////////////////////////////
	// ------------------------------------------------- Constructors
	/////////////////////////////////////////////////////////////////

	/**
	 * Empty (default) constructor.
	 */
		
	public MultipBinArrayRecombinator() 
	{
		super();
	}

	/**
	 * {@inheritDoc}
	 */
	
	@Override
	protected void prepareRecombination() 
	{
		ISpecies species = context.getSpecies();
		if (species instanceof MultipBinArraySpecies) {
			// Set individuals speciess
			this.species = (MultipBinArraySpecies) species;
		}
		else {
			throw new IllegalStateException("Invalid population species");
		}		
	}
	
}
