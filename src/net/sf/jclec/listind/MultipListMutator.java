package net.sf.jclec.listind;

import net.sf.jclec.ISpecies;
import net.sf.jclec.base.AbstractMutator;

/**
 * MultipListIndividual (and subclasses) specific mutator.  
 * 
 * @author Jose M. Moyano
 * @author Sebastian Ventura
 */

public abstract class MultipListMutator extends AbstractMutator  
{
	/**
	 * Generated by Eclipse
	 */
	private static final long serialVersionUID = 2071164648724732318L;
	
	/** Individual species (taked from execution context) */
	protected transient MultipListSpecies species;
	
	/** Genotype schema */ 	
	protected transient MultipListGenotype schema;
	
	/**
	 * Empty (default) constructor.
	 */
	public MultipListMutator() 
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
		if (species instanceof MultipListSpecies) {
			// Set individuals species
			this.species = (MultipListSpecies) species;
			// Sets genotype schema
			this.schema = this.species.getGenotypeSchema();
		}
		else {
			throw new IllegalStateException("Invalid species in context");
		}
	}
	
	@Override
	public boolean equals(Object other)
	{
		if (other instanceof MultipListMutator) {
			return true;
		}
		else {
			return false;
		}
	}
}