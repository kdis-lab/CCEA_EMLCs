package eaglet.recombinator;

import java.util.ArrayList;

import net.sf.jclec.binarray.BinArrayIndividual;
import net.sf.jclec.binarray.BinArrayRecombinator;

/**
 * Class implementing the random crossover
 * 
 * @author Jose M. Moyano
 *
 */
public class RandomCrossover extends BinArrayRecombinator {

	/**
	 * Serialization constant
	 */
	private static final long serialVersionUID = 8964405548093861255L;
	
	/**
	 *  Number of labels 
	 */
	private int numLabels;
	
	
	/**
	 * Constructor
	 */
	public RandomCrossover()
	{
		super();
	}
	
	/**
	 * Gets the number of labels
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
		if (other instanceof RandomCrossover) {
			return true;
		}
		else {
			return false;
		}
	}	
	
	
	@Override
	protected void recombineNext() {
		BinArrayIndividual p1 = (BinArrayIndividual) parentsBuffer.get(parentsCounter);
		BinArrayIndividual p2 = (BinArrayIndividual) parentsBuffer.get(parentsCounter+1);

		//Parents genotypes
		byte [] p1_genome = p1.getGenotype();
		byte [] p2_genome = p2.getGenotype();
		int gl = numLabels;
		
		//Stores the bits that are 1 in p1 and 0 in p2
		ArrayList<Integer> list1 = new ArrayList<Integer>();
		//Stores the bits that are 1 in p2 and 0 in p1
		ArrayList<Integer> list2 = new ArrayList<Integer>();
		
		//New active bits in son1
		ArrayList<Integer> listSon1 = new ArrayList<Integer>();
		//New active bits in son2
		ArrayList<Integer> listSon2 = new ArrayList<Integer>();
		
		for(int i=0; i<gl; i++)
		{
			if(p1_genome[i] == 1){
			}
			
			if(p1_genome[i] != p2_genome[i]){
				if(p1_genome[i] == 1){
					list1.add(i);
				}
				else{ //p2_genome[i] == 1
					list2.add(i);
				}
			}
		}
		
		//If more than one bit differs from one parent to other 
		if(list1.size() > 1){
			//Number of bits to swap
			int x = list1.size() / 2;
			int r;
			
			for(int i=0; i<x; i++){
				r = randgen.choose(0, list1.size());
				listSon1.add(list1.get(r));
				list1.remove(r);
				
				r = randgen.choose(0, list2.size());
				listSon2.add(list2.get(r));
				list2.remove(r);
			}
			listSon1.addAll(list2);
			listSon2.addAll(list1);
		}
		
		//Create sons genotypes
		byte [] s1_genome = new byte[gl];
		byte [] s2_genome = new byte[gl];
		
		System.arraycopy(p1_genome, 0, s1_genome, 0, gl);
		System.arraycopy(p2_genome, 0, s2_genome, 0, gl);
		
		for(int i=0; i<listSon1.size(); i++){
			s1_genome[listSon1.get(i)] = 1;
			s2_genome[listSon1.get(i)] = 0;
			
			s1_genome[listSon2.get(i)] = 0;
			s2_genome[listSon2.get(i)] = 1;
		}
		

		// Put sons in buffer
		sonsBuffer.add(species.createIndividual(s1_genome));
		sonsBuffer.add(species.createIndividual(s2_genome));
	}

}