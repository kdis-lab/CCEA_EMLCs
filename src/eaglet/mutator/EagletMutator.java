package eaglet.mutator;

import net.sf.jclec.binarray.BinArrayMutator;

/**
 * Class implementing an abstract mutator class which the rest will extend
 * 
 * @author Jose M. Moyano
 *
 */
public abstract class EagletMutator extends BinArrayMutator {

	/**
	 * Serialization constant
	 */
	private static final long serialVersionUID = -4724700530250317266L;
	/**
	 *  Number of labels 
	 */
	int numLabels;
	
	
	/**
	 * Constructor
	 */
	public EagletMutator()
	{
		super();
	}
	
	/**
	 * Sets the number of labels
	 * 
	 * @param numLabels Number of labels
	 */
	public void setNumLabels(int numLabels)
	{
		this.numLabels = numLabels;
	}
	
	@Override
	public boolean equals(Object other)
	{
		if (other instanceof EagletMutator) {
			return true;
		}
		else {
			return false;
		}
	}
	
}
