package net.sf.jclec.algorithm;

import java.util.ArrayList;
import java.util.List;

import net.sf.jclec.ISpecies;
import net.sf.jclec.algorithm.PopulationAlgorithm;
import net.sf.jclec.IProvider;
import net.sf.jclec.IEvaluator;
import net.sf.jclec.IConfigure;
import net.sf.jclec.IIndividual;
import net.sf.jclec.IPopulation;

import net.sf.jclec.util.random.IRandGen;
import net.sf.jclec.util.random.IRandGenFactory;


import org.apache.commons.lang.builder.EqualsBuilder;

import org.apache.commons.configuration.Configuration;
import org.apache.commons.configuration.ConfigurationRuntimeException;

/**
 * Algorithm that operates over a population. 
 * 
 * @author Sebastian Ventura
 * @author Jose M. Moyano
 */

@SuppressWarnings("serial")
public abstract class MultiPopulationAlgorithm extends PopulationAlgorithm
{

	/////////////////////////////////////////////////////////////////
	// ---------------------------- Internal variables (System state)
	/////////////////////////////////////////////////////////////////
	
	/* Sets for each subpopulation */
	
	/** Number of sub-populations */
	protected int numSubpop;
	
	/** Number of individuals in each subpopulation */
	protected int subpopSize;
		
	/** Actual individuals set */
	
	protected List<List<IIndividual>> bset;
	
	/** Individuals selected as parents */
	
	protected transient List<List<IIndividual>> pset;
	
	/** Individuals generated */
	
	protected transient List<List<IIndividual>> cset;
	
	/** Individuals to replace */
	
	protected transient List<List<IIndividual>> rset;

	/////////////////////////////////////////////////////////////////
	// ------------------------------------------------- Constructors
	/////////////////////////////////////////////////////////////////

	/**
	 * Empty (default) constructor
	 */
	
	public MultiPopulationAlgorithm() 
	{
		super();
	}

	/////////////////////////////////////////////////////////////////
	// ---------------------------------------- IPopulation interface
	/////////////////////////////////////////////////////////////////

	// System state
	
	public List<List<IIndividual>> getMultiInhabitants() 
	{
		return bset;
	}
	
	public List<IIndividual> getInhabitants(int i) 
	{
		return bset.get(i);
	}

	public final void setMultiInhabitants(List<List<IIndividual>> inhabitants)
	{
		this.bset = inhabitants;
	}
	
	public final void setInhabitants(List<IIndividual> inhabitants, int i)
	{
		this.bset.set(i, inhabitants);
	}

	public final int getNumSubpop() 
	{
		return numSubpop;
	}

	public final void setNumSubpop(int numSubpop) 
	{
		this.numSubpop = numSubpop;
	}

	// IConfigure interface
	
	/**
	 * Configuration parameters for BaseAlgorithm class are:
	 * 
	 * <ul>
	 * <li>
	 * <code>species: ISpecies (complex)</code></p>
	 * Individual species
	 * </li><li>
	 * <code>evaluator IEvaluator (complex)</code></p>
	 * Individuals evaluator
	 * </li><li>
	 * <code>population-size (int)</code></p>
	 * Population size
	 * </li><li>
	 * <code>max-of-generations (int)</code></p>
	 * Maximum number of generations
	 * </li>
	 * <li>
	 * <code>provider: IProvider (complex)</code></p>
	 * Individuals provider
	 * </li>
	 * </ul>
	 */
	
	public void configure(Configuration configuration)
	{
		// Call super.configure() method
		super.configure(configuration);
		
		// Population size
		int numSubpop = configuration.getInt("number-subpop");
		setNumSubpop(numSubpop);
	}
	
	
	// java.lang.Object methods

	@Override
	public boolean equals(Object other)
	{
		if (other instanceof MultiPopulationAlgorithm) {
			MultiPopulationAlgorithm cother = (MultiPopulationAlgorithm) other;
			EqualsBuilder eb = new EqualsBuilder();
			
			// Random generators factory
			eb.append(randGenFactory, cother.randGenFactory);
			// Individual species
			eb.append(species, cother.species);
			// Individuals evaluator
			eb.append(evaluator, cother.evaluator);
			// Population size
			eb.append(populationSize, cother.populationSize);
			// Max of generations
			eb.append(maxOfGenerations, cother.maxOfGenerations);
			// Individuals provider
			eb.append(provider, cother.provider);
			//Number of subpopulations
			eb.append(numSubpop, cother.numSubpop);
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

	// Execution methods
	
	/**
	 * Create individuals in population, evaluating before start rest
	 * of evolution
	 */
	
	protected void doInit() 
	{
		//Calculate individuals by subpopulation
		subpopSize = populationSize / numSubpop;
		
		for(int i=0; i<numSubpop; i++) {
			//Initialize each population
			bset.set(i, provider.provide(subpopSize));
			//bset = new ArrayList<List<IIndividual>>(numSubpop);
			pset = new ArrayList<List<IIndividual>>(numSubpop);
			cset = new ArrayList<List<IIndividual>>(numSubpop);
			rset = new ArrayList<List<IIndividual>>(numSubpop);
			
			// Evaluate individuals
			evaluator.evaluate(bset.get(i));
		}

		// Do Control
		doControl();
	}

	/**
	 * ... 
	 */
	
	protected void doIterate() 
	{
		generation++;
		// Do selection
		doSelection();
		// Do generation
		doGeneration();
		// Do replacement
		doReplacement();
		// Do update
		doUpdate();
		// Do control
		doControl();
		// Increments generation counter
	}
			
	/**
	 * Select individuals to be breeded. 
	 */
	
	protected abstract void doSelection();
	
	/**
	 * Generate new individuals from parents
	 */
	
	protected abstract void doGeneration();
	
	/**
	 * Select individuals to extinct in this generation.
	 */
	
	protected abstract void doReplacement();
	
	/**
	 * Update population individuals.
	 */
	
	protected abstract void doUpdate();
	
	/**
	 * Check if evolution is finished. Default implementation of this
	 * method performs the operations:
	 * 
	 * <ul>
	 * <li>
	 * If number of generations exceeds the maximum allowed, set the
	 * finished flag to true. Else, the flag remains false
	 * </li>
	 * <li>
	 * If number of evaluations exceeds the maximum allowed, set the
	 * finished flag to true. Else, the flag remains false
	 * </li>
	 * <li>
	 * If one individual has an  acceptable fitness, set the finished
	 * flag to true. Else, the flag remains false. 
	 * </li>
	 * </ul>
	 */
	
	protected void doControl()
	{
		// If maximum number of generations is exceeded, evolution is
		// finished
		if (generation >= maxOfGenerations) {
//			finished = true;
			state = FINISHED;
			return;
		}
		// If maximum number of evaluations is exceeded, evolution is
		// finished
		if (evaluator.getNumberOfEvaluations() > maxOfEvaluations) {
//			finished = true;
			state = FINISHED;
			return;
		}
		// If any individual in b has an acceptable fitness evolution
		// is finished
		/*
		for (IIndividual individual : bset) {
			if (individual.getFitness().isAcceptable()) {
//				finished = true;
				state = FINISHED;
				return;
			}
		}
		*/
	}	
}
