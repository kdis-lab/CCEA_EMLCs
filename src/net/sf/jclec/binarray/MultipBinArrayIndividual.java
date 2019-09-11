package net.sf.jclec.binarray;

import net.sf.jclec.IFitness;
import net.sf.jclec.IIndividual;

import net.sf.jclec.base.AbstractIndividual;

import java.util.Arrays;

import org.apache.commons.lang.builder.EqualsBuilder;

/**
 * Individual with a byte array as genotype.
 *  
 * @author Sebastian Ventura
 */

@SuppressWarnings("deprecation")
public class MultipBinArrayIndividual extends AbstractIndividual<byte[]> 
{
	/////////////////////////////////////////////////////////////////
	// --------------------------------------- Serialization constant
	/////////////////////////////////////////////////////////////////

	/** Generated by eclipse */
	
	private static final long serialVersionUID = 6227386750669278917L;

	/**
	 * Identifier of the sub-population
	 */
	protected int p = -1;
	
	/////////////////////////////////////////////////////////////////
	// ------------------------------------------------- Constructors
	/////////////////////////////////////////////////////////////////
	
	/**
	 * Empty constructor
	 */
	
	public MultipBinArrayIndividual() 
	{
		super();
	}

	/**
	 * Constructor that sets individual genotype.
	 * 
	 * @param genotype Individual genotype
	 */
	
	public MultipBinArrayIndividual(byte[] genotype) 
	{
		super(genotype);
	}
	
	/**
	 * Constructor that sets the individual genotype and the identifier of sub-population
	 * 
	 * @param genotype Individual genotype
	 * @param p Identifier of subpopulation
	 */
	public MultipBinArrayIndividual(byte[] genotype, int p) 
	{
		super(genotype);
		this.p = p;
	}

	/**
	 * Constructor that sets individual genotype and fitness.
	 * 
	 * @param genotype Individual genotype
	 * @param fitness  Individual fitness
	 */
	
	public MultipBinArrayIndividual(byte[] genotype, IFitness fitness) 
	{
		super(genotype, fitness);
	}
	
	
	/**
	 * Constructor that sets individual genotype, fitness, and subpop identifier.
	 * 
	 * @param genotype Individual genotype
	 * @param fitness  Individual fitness
	 * @param p Identifier of subpopulation
	 */
	
	public MultipBinArrayIndividual(byte[] genotype, IFitness fitness, int p) 
	{
		super(genotype, fitness);
		this.p = p;
	}
	
	
	/**
	 * Sets the identifier of subpopulation
	 * 
	 * @param p Sub-population identifier
	 */
	public void setSubpop(int p) {
		this.p = p;
	}

	/////////////////////////////////////////////////////////////////
	// ----------------------------------------------- Public methods
	/////////////////////////////////////////////////////////////////
		
	// IIndividual methods
		
	/**
	 * {@inheritDoc}
	 */
		
	public IIndividual copy() 
	{
		// Genotype length
		int gl = genotype.length;
		// Allocate a copy of genotype
		byte [] gother = new byte[genotype.length];
		// Copy genotype
		System.arraycopy(genotype, 0, gother, 0, gl);
		// Create new individuals, then return it
		if (fitness != null) {
			return new MultipBinArrayIndividual(gother, fitness.copy(), p);			
		}
		else {
			return new MultipBinArrayIndividual(gother, p);			
		}
	}

	/**
	 * BinArrayIndividual uses the Hamming distance for setting 
	 * differences between individuals - at genotype level.
	 * 
	 * {@inheritDoc}
	 */
	
	public double distance(IIndividual other) 
	{
		// Other genotype
		byte [] gother = ((MultipBinArrayIndividual) other).genotype;
		// Setting Hamming distance
		int distance = 0;
		int gl = genotype.length;
		for (int i=0; i<gl; i++) {
			if (genotype[i] != gother[i])
				distance++;
		}	
		// Returns hamming distance
		return (double) distance;
	}

	// java.lang.Object methods
	
	/**
	 * {@inheritDoc}
	 */
	
	@Override
	public boolean equals(Object other) 
	{
		if (other instanceof MultipBinArrayIndividual) {
			MultipBinArrayIndividual baother = (MultipBinArrayIndividual) other;
			EqualsBuilder eb = new EqualsBuilder();
			eb.append(genotype, baother.genotype);
			eb.append(fitness, baother.fitness);
			eb.append(p, baother.p);
			return eb.isEquals();
		}
		else {
			return false;
		}
	}
	
	public String toString() {
		return new String(p + Arrays.toString(genotype));
	}
}
