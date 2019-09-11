package net.sf.jclec.binarray;

import net.sf.jclec.ISpecies;

import net.sf.jclec.base.AbstractMutator;

/**
 * BinArrayIndividual (and subclasses) specific mutator.  
 * 
 * @author Sebastian Ventura
 */

public abstract class MultipBinArrayMutator extends BinArrayMutator  
{
	/** Individual species (taked from execution context) */
	
	protected transient MultipBinArraySpecies species;
	
	/**
	 * Empty (default) constructor.
	 */
	
	public MultipBinArrayMutator() 
	{
		super();
	}

	/////////////////////////////////////////////////////////////////
	// ----------------------------------------------- Public methods
	/////////////////////////////////////////////////////////////////
	
	// AbstractMutator methods
	
	/**
	 * {@inheritDoc}
	 */
	
	@Override
	protected void prepareMutation() 
	{
		ISpecies species = context.getSpecies();
		if (species instanceof MultipBinArraySpecies) {
			// Set individuals species
			this.species = (MultipBinArraySpecies) species;
			// Sets genotype schema
			this.schema = this.species.getGenotypeSchema();
		}
		else {
			throw new IllegalStateException("Invalid species in context");
		}
	}

	/* 
	 * Este mtodo fija el schema que vamos a utilizar para mutar los genotipos
	 * de los nuevos individuos. Para ello, asegura que el objeto species que
	 * representa a los individuos de la poblacin es de tipo IBinArraySpecies.
	 * En caso negativo, lanza una excepcin.
	 */
}
