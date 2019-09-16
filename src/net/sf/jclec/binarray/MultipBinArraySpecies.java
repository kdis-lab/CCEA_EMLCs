package net.sf.jclec.binarray;

import net.sf.jclec.ISpecies;

/**
 * Abstract implementation for IBinArraySpecies.
 * 
 * This class  contains a byte array  that contains the genotype schema for all
 * represented individuals. This schema can be set in a subclass of this or can 
 * be calculated from other problem information.
 * 
 * It also allows the use of multiple subpopulations
 * 
 * @author Sebastian Ventura
 * @author Jose M. Moyano
 * 
 * @see MultipBinArrayIndividualSpecies
 */

@SuppressWarnings("serial")
public abstract class MultipBinArraySpecies implements ISpecies
{
	/////////////////////////////////////////////////////////////////
	// ----------------------------------------------------Properties
	/////////////////////////////////////////////////////////////////
	
	/** Genotype schema */
	protected byte [] genotypeSchema;
	
	/**
	 * Identifier of subpopulation
	 */
	protected int p;
	
	/////////////////////////////////////////////////////////////////
	// ------------------------------------------------- Constructors
	/////////////////////////////////////////////////////////////////
	
	/**
	 * Empty constructor
	 */
	public MultipBinArraySpecies() 
	{
		super();
	}

	/////////////////////////////////////////////////////////////////
	// ----------------------------------------------- Public methods
	/////////////////////////////////////////////////////////////////
	
	// Factory method
	
	/**
	 * Factory method.
	 * 
	 * @param genotype Individual genotype.
	 * 
	 * @return A new instance of represented class
	 */
	public abstract MultipBinArrayIndividual createIndividual(byte [] genotype);	
	
	/**
	 * Factory method.
	 * 
	 * @param genotype Individual genotype
	 * @param p Index of subpopulation
	 * @return A new instance of represented class
	 */
	public abstract MultipBinArrayIndividual createIndividual(byte [] genotype, int p);	

	// Genotype information

	/**
	 * Informs about individual genotype length.
	 * 
	 * @return getGenotypeSchema().length
	 */
	public int getGenotypeLength() 
	{
		return genotypeSchema.length;
	}

	/**
	 * @return This genotype schema
	 */
	public byte[] getGenotypeSchema() 
	{
		return genotypeSchema;
	}
	
	/**
	 * Get identifier of subpopulation
	 * 
	 * @return Identifier of subpopulation
	 */
	public int getSubpopId() {
		return p;
	}
}
