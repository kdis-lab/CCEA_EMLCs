package coeaglet.recombinator;

import java.util.ArrayList;
import java.util.Collections;

import net.sf.jclec.listind.MultipListGenotype;
import net.sf.jclec.listind.MultipListIndividual;
import net.sf.jclec.listind.MultipListRecombinator;

/**
 * Class implementing the crossover operator
 * 
 * @author Jose M. Moyano
 *
 */
public class Crossover extends MultipListRecombinator {

	/** Serialization constant */
	private static final long serialVersionUID = 8964405548093861255L;
	
	/**
	 * Constructor
	 */
	public Crossover()
	{
		super();
	}
	
	@Override
	public boolean equals(Object other)
	{
		if (other instanceof Crossover) {
			return true;
		}
		else {
			return false;
		}
	}	
	
	
	/**
	 * Recombine next individuals
	 * 
	 */
	protected void recombineNext() {
		MultipListIndividual p1 = (MultipListIndividual) parentsBuffer.get(parentsCounter);
		MultipListIndividual p2 = (MultipListIndividual) parentsBuffer.get(parentsCounter+1);

		//Get copies of the list of integers
		ArrayList<Integer> set1 = new ArrayList<Integer>(p1.getGenotype().genotype);
		ArrayList<Integer> set2 = new ArrayList<Integer>(p2.getGenotype().genotype);
		ArrayList<Integer> repeated = new ArrayList<Integer>();
		
		//Store which indexes are repeated in both individuals
		for(int i : set1) {
			if(set2.contains(i)) {
				repeated.add(i);
			}
		}
		
		//If less than 1 genes are different, don't cross (it does not make sense)
		if(repeated.size() >= (set1.size()-1)) {
			//Just put parents in the buffer to maintain size of cset
			sonsBuffer.add(p1);
			sonsBuffer.add(p2);
		}
		else {
			//If there are repeated genes, remove from both sets
			//The (Integer) is to make sure we are removing the object and not the index
			if(repeated.size() > 0) {
				for(int i : repeated) {
					set1.remove(new Integer(i));
					set2.remove(new Integer(i));
				}
			}
			
			//Shuffle sets
			Collections.shuffle(set1);
			Collections.shuffle(set2);
			
			//Create new sets with one half of each of them; and add repeated
			ArrayList<Integer> newSet1 = new ArrayList<Integer>();
			ArrayList<Integer> newSet2 = new ArrayList<Integer>();
			for(int i=0; i<set1.size()/2; i++) {
				newSet1.add(set1.get(i));
				newSet2.add(set2.get(i));
			}
			for(int i=set1.size()/2; i<set1.size(); i++) {
				newSet1.add(set2.get(i));
				newSet2.add(set1.get(i));
			}
			for(int i : repeated) {
				newSet1.add(i);
				newSet2.add(i);
			}
			
			//Sort new sets
			Collections.sort(newSet1);
			Collections.sort(newSet2);
			
			//Create new individuals with new sets
			MultipListIndividual s1 = new MultipListIndividual(new MultipListGenotype(p1.getGenotype().subpop, newSet1));
			MultipListIndividual s2 = new MultipListIndividual(new MultipListGenotype(p2.getGenotype().subpop, newSet2));
			
			// Put sons in buffer
			sonsBuffer.add(s1);
			sonsBuffer.add(s2);
		}
	}
	

}