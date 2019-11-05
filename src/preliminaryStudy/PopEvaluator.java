package preliminaryStudy;

import java.util.Hashtable;
import java.util.List;

import mulan.classifier.MultiLabelLearner;
import mulan.data.MultiLabelInstances;
import net.sf.jclec.listind.MultipListIndividual;

/** 
 * @author Jose M. Moyano
 *
 */
public class PopEvaluator {
	
	public void evaluatePop(List<List<MultipListIndividual>> inds, MultiLabelInstances[] train, MultiLabelInstances[] validation,
			MultiLabelLearner learner, Hashtable<String, MultiLabelLearner> tableClassifiers) {
		try {
			for(int p=0; p<inds.size(); p++) {
				//System.out.println("\t\t\tEvaluating subpop " + p + " valdiation");
				
				MemberEvaluator memberEval = new MemberEvaluator();
				memberEval.evaluateInds(inds.get(p), train[p], validation[p], learner, tableClassifiers);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}
}
