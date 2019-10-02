package preliminaryStudy;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Collections;



import net.sf.jclec.IFitness;
import net.sf.jclec.IIndividual;
import net.sf.jclec.ISystem;
import net.sf.jclec.binarray.MultipBinArrayIndividual;
import net.sf.jclec.selector.DeterministicSelector;

import org.apache.commons.configuration.Configuration;

/**
 * Better individuals selector.
 * 
 * @author Sebastian Ventura
 */

public class BettersSelector2 extends DeterministicSelector 
{
	/////////////////////////////////////////////////////////////////
	// --------------------------------------- Serialization constant
	/////////////////////////////////////////////////////////////////

	/** Generated by Eclipse */
	
	private static final long serialVersionUID = -697152908546090334L;
	
	protected transient List<MultipBinArrayIndividual> actsrc;
	
	/////////////////////////////////////////////////////////////////
	// ------------------------------------------- Internal variables
	/////////////////////////////////////////////////////////////////

	/** Compare individuals */
	
	protected transient Comparator<IIndividual> individualsComparator =  
		new Comparator<IIndividual> () 
		{
			/**
			 * {@inheritDoc} 
			 */
		
			public int compare(IIndividual arg0, IIndividual arg1) {
				return fitnessComparator.compare(arg0.getFitness(), arg1.getFitness());
			}
		};
	
	/** Fitness comparator (taken from context) */
	
	protected transient Comparator<IFitness> fitnessComparator;
	
	/** Auxiliary list */
	
	protected transient ArrayList<MultipBinArrayIndividual> auxList = new ArrayList<MultipBinArrayIndividual> ();

	/////////////////////////////////////////////////////////////////
	// ------------------------------------------------- Constructors
	/////////////////////////////////////////////////////////////////

	/**
	 * Empty constructor.
	 */
	
	public BettersSelector2() 
	{
		super();
	}

	public BettersSelector2(ISystem context) 
	{
		super();
		contextualize(context);
	}

	/////////////////////////////////////////////////////////////////
	// ----------------------------------------------- Public methods
	/////////////////////////////////////////////////////////////////
	
	// IConfigure interface

	public final void configure(Configuration settings) 
	{
		// Do nothing
	}

	// java.lang.Object methods

	@Override
	public boolean equals(Object other)
	{
		return (other instanceof BettersSelector2);
	}

	/////////////////////////////////////////////////////////////////
	// -------------------------------------------- Protected methods
	/////////////////////////////////////////////////////////////////

	// AbstractSelector methods

	/**
	 * Prepare selection consists on take fitnesses comparator (to
	 * compare individuals by fitness). Then, copy all source inds in
	 * a temporary list
	 * 
	 * {@inheritDoc} 
	 */
	
	@Override
	protected void prepareSelection() 
	{
		// Clear auxiliary list
		auxList.clear();
		// Set fitness comparator
		MemberEvaluator eval = new MemberEvaluator();
		fitnessComparator = eval.getComparator();
		// Puts source individuals in auxlist
		for (MultipBinArrayIndividual ind : actsrc) auxList.add(ind);
	}

	/**
	 * This method take best individual in temporary list, then remove
	 * and return it.  
	 * 
	 * {@inheritDoc}
	 */
	
	@Override
	protected MultipBinArrayIndividual selectNext() 
	{
		// Security mechanism
		if (auxList.isEmpty()) prepareSelection();
		// Select actual best
		MultipBinArrayIndividual best = Collections.max(auxList, individualsComparator);
		// Extract best from auxlist
		auxList.remove(best);
		// Return best individual
		return best;	
	}
	
	public List<MultipBinArrayIndividual> selectMultip(List<MultipBinArrayIndividual> src, int nofsel) 
	{
		return selectMultip(src, nofsel, true);
	}
	
	public List<MultipBinArrayIndividual> selectMultip(List<MultipBinArrayIndividual> src, int nofsel, boolean repeat) 
	{
		// Sets source set and actsrcsz
		actsrc = src; actsrcsz = src.size();
		// Prepare selection process
		prepareSelection();
		// Performs selection of n individuals
		ArrayList<MultipBinArrayIndividual> result = new ArrayList<MultipBinArrayIndividual>();
		for (int i=0; i<nofsel; i++) {
			MultipBinArrayIndividual selected = selectNext(); 
			if (!repeat) {
				while (result.contains(selected)) {
					selected = selectNext();
				}
			}
			result.add(selected);		
		}		
		// Returns selection
		return result;
	}
}