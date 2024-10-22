package net.sf.jclec.algorithm.classic;

import net.sf.jclec.IMutator;
import net.sf.jclec.IRecombinator;
import net.sf.jclec.ISelector;
import net.sf.jclec.IConfigure;
import net.sf.jclec.IIndividual;

import net.sf.jclec.base.FilteredMutator;
import net.sf.jclec.base.FilteredRecombinator;
import net.sf.jclec.listind.MultipListGenotype;
import net.sf.jclec.listind.MultipListIndividual;
import net.sf.jclec.algorithm.MultiPopulationAlgorithm;

import org.apache.commons.lang.builder.EqualsBuilder;

import coeaglet.algorithm.MultipAbstractParallelEvaluator;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.configuration.Configuration;
import org.apache.commons.configuration.ConfigurationRuntimeException;

/**
 * <strong><u>S</u></strong>imple <strong><u>G</u></strong>enerational algorithm.
 * 
 * It is implemented to allow multiple subpopulations.
 * 
 * @author Sebastian Ventura
 * @author Jose M. Moyano
 */

public class MultiSG extends MultiPopulationAlgorithm 
{
	/////////////////////////////////////////////////////////////////
	// --------------------------------------- Serialization constant
	/////////////////////////////////////////////////////////////////

	/** Generated by Eclipse */	
	private static final long serialVersionUID = -2649346083463795286L;

	/////////////////////////////////////////////////////////////////
	// --------------------------------------------------- Properties
	/////////////////////////////////////////////////////////////////
	
	/** Parents selector */	
	protected ISelector parentsSelector;
	
	/** Individuals mutator */
	protected FilteredMutator mutator;

	/** Individuals recombinator */
	protected FilteredRecombinator recombinator;
	
	/////////////////////////////////////////////////////////////////
	// ------------------------------------------------- Constructors
	/////////////////////////////////////////////////////////////////

	/**
	 * Empty (default) constructor
	 */
	public MultiSG() 
	{
		super();
	}

	/////////////////////////////////////////////////////////////////
	// ----------------------------------------------- Public methods
	/////////////////////////////////////////////////////////////////

	// Getting and setting properties
	
	/**
	 * Access to parents selector
	 * 
	 * @return Parents selector
	 */
	public ISelector getParentsSelector() 
	{
		return parentsSelector;
	}

	/**
	 * Sets the parents selector.
	 * 
	 * @param parentsSelector New parents selector
	 */
	public void setParentsSelector(ISelector parentsSelector) 
	{
		// Sets the parents selector
		this.parentsSelector = parentsSelector;
		// Contextualize parents selector
		parentsSelector.contextualize(this);
	}

	// Generation plan
	
	/**
	 * Access to parents recombinator
	 * 
	 * @return Actual parents recombinator
	 */
	public IRecombinator getRecombinator() 
	{
		return recombinator.getDecorated();
	}

	/**
	 * Sets the parents recombinator.
	 * 		
	 * @param recombinator New parents recombinator
	 */
	public void setRecombinator(IRecombinator recombinator) 
	{
		if (this.recombinator == null) {
			this.recombinator = new FilteredRecombinator (this);
		}
		this.recombinator.setDecorated(recombinator);
	}
	
	/**
	 * Access to recombination probability
	 * 
	 * @return Recombination probability
	 */
	public double getRecombinationProb()
	{
		return recombinator.getRecProb();
	}
	
	/**
	 * Set recombination probability
	 * 
	 * @param recProb Recombination probability
	 */
	public void setRecombinationProb(double recProb)
	{
		if (this.recombinator == null) {
			this.recombinator = new FilteredRecombinator (this);
		}
		recombinator.setRecProb(recProb);
	}
	
	/**
	 * Access to individuals mutator.
	 * 
	 * @return Individuals mutator
	 */
	public IMutator getMutator() 
	{
		return mutator.getDecorated();
	}

	/**
	 * Set individuals mutator.
	 * 
	 * @param mutator Individuals mutator
	 */
	public void setMutator(IMutator mutator) 
	{
		if (this.mutator == null) {
			this.mutator = new FilteredMutator (this);
		}
		this.mutator.setDecorated(mutator);
	}
	
	/**
	 * Access to mutation probability
	 * 
	 * @return Mutation probability
	 */
	public double getMutationProb()
	{
		return mutator.getMutProb();
	}
	
	/**
	 * Set mutation probability
	 * 
	 * @param mutProb Mutation probability
	 */
	public void setMutationProb(double mutProb)
	{
		if (this.mutator == null) {
			this.mutator = new FilteredMutator(this);
		}
		mutator.setMutProb(mutProb);
	}

	// IConfigure interface
	
	/**
	 * Configuration method.
	 * 
	 * Configuration parameters for BaseAlgorithm class are:
	 * 
	 * <ul>
	 * <li>
	 * <code>species: ISpecies (complex)</code></p>
	 * Individual species
	 * </li><li>
	 * <code>evaluator: IEvaluator (complex)</code></p>
	 * Individuals evaluator
	 * </li><li>
	 * <code>population-size: int</code></p>
	 * Population size
	 * </li><li>
	 * <code>max-of-generations: int</code></p>
	 * Maximum number of generations
	 * </li>
	 * <li>
	 * <code>provider: IProvider (complex)</code></p>
	 * Individuals provider
	 * </li>
	 * <li>
	 * <code>parents-selector: ISelector (complex)</code> 
	 * </li>
	 * <li>
	 * <code>recombinator: (complex)</code>
	 * 		<ul>
	 * 		<li>
	 * 		<code>recombinator.decorated: IRecombinator (complex)</code></p>
	 * 		Recombination operator
	 * 		</li><li>
	 * 		<code>recombinator.recombination-prob double</code></p>
	 * 		Recombination probability
	 * 		</li>
	 * 		</ul> 
	 * </li>
	 * <li>
	 * <code>mutator: (complex)</code> 
	 * 		<ul>
	 * 		<li>
	 * 		<code>mutator.decorated: IMutator (complex) </code></p>
	 * 		Mutation operator
	 * 		</li><li>
	 * 		<code>mutator.mutation-prob double</code></p>
	 * 		Mutation probability
	 * 		</li>
	 * 		</ul> 
	 * </li>
	 * </ul>
	 */
	@SuppressWarnings("unchecked")
	public void configure(Configuration configuration)
	{
		// Call super.configure() method
		super.configure(configuration);
		// Parents selector
		try {
			// Selector classname
			String parentsSelectorClassname = 
				configuration.getString("parents-selector[@type]");
			// Species class
			Class<? extends ISelector> parentsSelectorClass = 
				(Class<? extends ISelector>) Class.forName(parentsSelectorClassname);
			// Species instance
			ISelector parentsSelector = parentsSelectorClass.newInstance();
			// Configure species if necessary
			if (parentsSelector instanceof IConfigure) {
				// Extract species configuration
				Configuration parentsSelectorConfiguration = configuration.subset("parents-selector");
				// Configure species
				((IConfigure) parentsSelector).configure(parentsSelectorConfiguration);
			}
			// Set species
			setParentsSelector(parentsSelector);
		} 
		catch (ClassNotFoundException e) {
			throw new ConfigurationRuntimeException("Illegal parents selector classname");
		} 
		catch (InstantiationException e) {
			throw new ConfigurationRuntimeException("Problems creating an instance of parents selector", e);
		} 
		catch (IllegalAccessException e) {
			throw new ConfigurationRuntimeException("Problems creating an instance of parents selector", e);
		}
		// Recombinator 
		try {
			// Recombinator classname
			String recombinatorClassname = 
				configuration.getString("recombinator[@type]");
			// Recombinator class
			Class<? extends IRecombinator> recombinatorClass = 
				(Class<? extends IRecombinator>) Class.forName(recombinatorClassname);
			// Recombinator instance
			IRecombinator recombinator = recombinatorClass.newInstance();
			// Configure recombinator if necessary
			if (recombinator instanceof IConfigure) {
				// Extract recombinator configuration
				Configuration recombinatorConfiguration = configuration.subset("recombinator");
				// Configure species
				((IConfigure) recombinator).configure(recombinatorConfiguration);
			}
			// Set species
			setRecombinator(recombinator);
		} 
		catch (ClassNotFoundException e) {
			throw new ConfigurationRuntimeException("Illegal recombinator classname");
		} 
		catch (InstantiationException e) {
			throw new ConfigurationRuntimeException("Problems creating an instance of recombinator", e);
		} 
		catch (IllegalAccessException e) {
			throw new ConfigurationRuntimeException("Problems creating an instance of recombinator", e);
		}
		// Recombination probability 
		double recProb = configuration.getDouble("recombinator[@rec-prob]");
		setRecombinationProb(recProb);
		// Mutator 
		String mutatorClassname = null;
		try {
			// Mutator classname
			mutatorClassname = 
				configuration.getString("mutator[@type]");
			// Recombinator class
			Class<? extends IMutator> mutatorClass = 
				(Class<? extends IMutator>) Class.forName(mutatorClassname);
			// Recombinator instance
			IMutator mutator = mutatorClass.newInstance();
			// Configure recombinator if necessary
			if (mutator instanceof IConfigure) {
				// Extract recombinator configuration
				Configuration mutatorConfiguration = configuration.subset("mutator");
				// Configure species
				((IConfigure) mutator).configure(mutatorConfiguration);
			}
			// Set mutator
			setMutator(mutator);
		} 
		catch (ClassNotFoundException e) {
			throw new ConfigurationRuntimeException("Illegal mutator classname");
		} 
		catch (InstantiationException e) {
			throw new ConfigurationRuntimeException("Problems creating an instance of mutator " + mutatorClassname, e);
		} 
		catch (IllegalAccessException e) {
			throw new ConfigurationRuntimeException("Problems creating an instance of mutator", e);
		}
		// Mutation probability 
		double mutProb = configuration.getDouble("mutator[@mut-prob]");
		setMutationProb(mutProb);
	}	
	
	// java.lang.Object methods

	@Override
	public boolean equals(Object other)
	{
		if (other instanceof SG) {
			SG cother = (SG) other;
			EqualsBuilder eb = new EqualsBuilder();
			// Call super method
			eb.appendSuper(super.equals(other));
			// Parents selector
			eb.append(parentsSelector, cother.parentsSelector);
			// Mutator
			eb.append(mutator, cother.mutator);
			// Recombinator
			eb.append(recombinator, cother.recombinator);
			// Return test result
			return eb.isEquals();
		}
		else {
			return false;
		}
	}

	
	/////////////////////////////////////////////////////////////////
	// -------------------------------------------- Protected methods
	/////////////////////////////////////////////////////////////////
	
	@Override
	protected void doSelection() 
	{
		pset = new ArrayList<List<IIndividual>>(numSubpop);

		for(int p=0; p<numSubpop; p++) {
			pset.add(p, parentsSelector.select(bset.get(p), subpopSize));
		}
	}

	@Override
	protected void doGeneration() 
	{
		cset = new ArrayList<List<IIndividual>>(numSubpop);
		for(int i=0; i<numSubpop; i++) {
			// Recombine parents
			cset.add(i, recombinator.recombine(pset.get(i)));
			
			// Add non-recombined inds. 
			// These individuals are references to existent individuals 
			// (elements of bset) so we make a copy of them
			for (IIndividual ind : recombinator.getSterile()) 
				cset.get(i).add(ind.copy());
			
			// Mutate filtered inds
			cset.set(i, mutator.mutate(cset.get(i)));
			
			// Add non-mutated inds. 
			// These individuals don't have to be copied, because there
			// are original individuals (not references)
			for (IIndividual ind : mutator.getSterile()) {
				cset.get(i).add(ind);
			}
			// Evaluate all new individuals
			//evaluator.evaluate(cset.get(i));	
		}
		((MultipAbstractParallelEvaluator)evaluator).evaluateMultip(cset);

		
	}

	@Override
	protected void doReplacement() 
	{
		rset = new ArrayList<List<IIndividual>>(numSubpop);
		for(int p=0; p<numSubpop; p++) {
			rset.add(p, bset.get(p));
		}
	}

	@Override
	protected void doUpdate() 
	{
		for(int p=0; p<numSubpop; p++) {
			bset.get(p).clear();
			bset.get(p).addAll(cset.get(p));

			pset.get(p).clear();
			rset.get(p).clear();
			cset.get(p).clear();
		}
	}
	
	@Override
	protected void doCommunication() 
	{
		for(int p=0; p<numSubpop; p++) {
			MultipListIndividual best = (MultipListIndividual)bset.get(p).get(0).copy();
			int r;
			do {
				r = randgen.choose(0, numSubpop);
			}while(r == best.getGenotype().subpop);
			
			best.setGenotype(new MultipListGenotype(r, best.getGenotype().genotype));
			bset.get(r).add(best);
		}
	}
}
