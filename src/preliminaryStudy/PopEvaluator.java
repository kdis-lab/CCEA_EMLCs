package preliminaryStudy;

import java.util.Hashtable;
import java.util.List;

import mulan.classifier.MultiLabelLearner;
import mulan.data.MultiLabelInstances;
import net.sf.jclec.binarray.MultipBinArrayIndividual;

/** 
 * @author Jose M. Moyano
 *
 */
public class PopEvaluator {

	public void evaluatePop(List<List<MultipBinArrayIndividual>> inds, MultiLabelInstances[] train,
			MultiLabelLearner learner, Hashtable<String, MultiLabelLearner> tableClassifiers) {
		try {
			for(int p=0; p<inds.size(); p++) {
				//System.out.println("\t\t\tEvaluating subpop " + p + " train");
				
				MemberEvaluator memberEval = new MemberEvaluator();
				memberEval.evaluateInds(inds.get(p), train[p], null, learner, tableClassifiers);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}
	
	public void evaluatePop(List<List<MultipBinArrayIndividual>> inds, MultiLabelInstances[] train, MultiLabelInstances[] validation,
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
