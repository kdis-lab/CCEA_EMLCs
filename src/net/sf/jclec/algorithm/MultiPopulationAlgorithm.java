package net.sf.jclec.algorithm;

import java.util.ArrayList;
import java.util.List;

import net.sf.jclec.algorithm.PopulationAlgorithm;
import net.sf.jclec.listind.MultipListCreator;
import net.sf.jclec.util.random.IRandGen;
import net.sf.jclec.IIndividual;

import org.apache.commons.lang.builder.EqualsBuilder;

import org.apache.commons.configuration.Configuration;

/**
 * Algorithm that operates over multiple populations. 
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
	
	/** Number of generations between communications */
	protected int generationsComm;
		
	/** Actual individuals set */
	protected List<List<IIndividual>> bset;
	
	/** Individuals selected as parents */
	protected transient List<List<IIndividual>> pset;
	
	/** Individuals generated */
	protected transient List<List<IIndividual>> cset;
	
	/** Individuals to replace */
	protected transient List<List<IIndividual>> rset;
	
	/** Random number generator */
	protected IRandGen randgen;

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
	
	/**
	 * Get list of bsets (one bset per subpopulation)
	 * 
	 * @return List of subpopulations (bsets)
	 */
	public List<List<IIndividual>> getMultiInhabitants() 
	{
		return bset;
	}
	
	/**
	 * Get individuals for a given subpopulation
	 * 
	 * @param p Index of subpopulation
	 * @return List of individuals of the subpopulation
	 */
	public List<IIndividual> getInhabitants(int p) 
	{
		return bset.get(p);
	}

	/**
	 * Set subspopulations
	 * 
	 * @param inhabitants List of lists with the individuals of each subpopulation
	 */
	public final void setMultiInhabitants(List<List<IIndividual>> inhabitants)
	{
		this.bset = inhabitants;
	}
	
	/**
	 * Set individuals for a given subpopulation
	 * 
	 * @param inhabitants List of individuals 
	 * @param p Index of subpopulation
	 */
	public final void setInhabitants(List<IIndividual> inhabitants, int p)
	{
		this.bset.set(p, inhabitants);
	}

	/**
	 * Get the number of subpopulations in the environment
	 * 
	 * @return Number of subpopulations
	 */
	public final int getNumSubpop() 
	{
		return numSubpop;
	}

	/**
	 * Set the number of subpopulations
	 * 
	 * @param numSubpop Number of subpopulations
	 */
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
		
		generationsComm = configuration.getInt("ngenerations-comm");
		
		randgen = randGenFactory.createRandGen();
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
		
		bset = new ArrayList<List<IIndividual>>(numSubpop);
		pset = new ArrayList<List<IIndividual>>(numSubpop);
		cset = new ArrayList<List<IIndividual>>(numSubpop);
		rset = new ArrayList<List<IIndividual>>(numSubpop);
		
		for(int p=0; p<numSubpop; p++) {
			//Initialize each population
			((MultipListCreator) provider).setSubpopId(p);
			List<IIndividual> prov = provider.provide(subpopSize);
			bset.add(p, prov);
			
			// Evaluate individuals
			evaluator.evaluate(bset.get(p));
		}

		// Do Control
		doControl();
	}

	@Override
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
	 * Communicate subpopulations
	 */
	protected abstract void doCommunication();
	
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
		if ((generation % generationsComm) == 0 && generation > 0) {
			doCommunication();
		}
		
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
		// is finished --> NOT CONTROLLED
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
