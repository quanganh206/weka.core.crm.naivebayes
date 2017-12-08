# weka.core.crm.naivebayes

Use Weka and apply basic modeling with the Naive Bayes to obtain our own baseline AUC scores.

```bash
Start processing...
Naive Bayes
	churn: 0.5903249542948283
	appetency: 0.6355512847791818
	up-sell: 0.6714814959494408
	overall: 0.6324525783411503

Processing Time: PT50.935S
End processing...
```

Sorry I working on a weak machine :D

```bash
Start processing...
[weka.classifiers.bayes.NaiveBayes , weka.classifiers.lazy.IBk -K 1 -W 0 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"", weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump, weka.classifiers.meta.LogitBoost -P 100 -L -1.7976931348623157E308 -H 1.0 -Z 3.0 -O 1 -E 1 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump, weka.classifiers.trees.J48 -S -C 0.25 -B -M 2, weka.classifiers.trees.J48 -S -C 0.25 -B -M 2 -A]
0.7131504734086965 weka.classifiers.meta.LogitBoost -P 100 -L -1.7976931348623157E308 -H 1.0 -Z 3.0 -O 1 -E 1 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.6992882586789329 weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.5879078161632199 weka.classifiers.bayes.NaiveBayes 
0.5112969219535686 weka.classifiers.lazy.IBk -K 1 -W 0 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\""
0.49471241390110415 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2
0.49437992594897096 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2 -A
0.7113813324196618 weka.classifiers.meta.LogitBoost -P 100 -L -1.7976931348623157E308 -H 1.0 -Z 3.0 -O 1 -E 1 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.6904627533595113 weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.5904419026069282 weka.classifiers.bayes.NaiveBayes 
0.509840023372239 weka.classifiers.lazy.IBk -K 1 -W 0 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\""
0.4947653122163259 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2 -A
0.49423011167532305 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2
0.7015862542478908 weka.classifiers.meta.LogitBoost -P 100 -L -1.7976931348623157E308 -H 1.0 -Z 3.0 -O 1 -E 1 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.6932386521450505 weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.5803101670672061 weka.classifiers.bayes.NaiveBayes 
0.5083031711915481 weka.classifiers.lazy.IBk -K 1 -W 0 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\""
0.5049744804200743 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2
0.5047868016058606 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2 -A
0.7088870691430802 weka.classifiers.meta.LogitBoost -P 100 -L -1.7976931348623157E308 -H 1.0 -Z 3.0 -O 1 -E 1 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.6950353496785067 weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.5737245780085075 weka.classifiers.bayes.NaiveBayes 
0.5074348096488958 weka.classifiers.lazy.IBk -K 1 -W 0 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\""
0.4963018916589318 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2
0.49580215361983243 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2 -A
0.7112976780804746 weka.classifiers.meta.LogitBoost -P 100 -L -1.7976931348623157E308 -H 1.0 -Z 3.0 -O 1 -E 1 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.6928596746224308 weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.5872234533969624 weka.classifiers.bayes.NaiveBayes 
0.511183347909275 weka.classifiers.lazy.IBk -K 1 -W 0 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\""
0.5033324443446656 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2
0.5026881503359302 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2 -A
0.8166405915749785 weka.classifiers.meta.LogitBoost -P 100 -L -1.7976931348623157E308 -H 1.0 -Z 3.0 -O 1 -E 1 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.8138114941749661 weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.5866071622635725 weka.classifiers.bayes.NaiveBayes 
0.5162068813646961 weka.classifiers.lazy.IBk -K 1 -W 0 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\""
0.4885604204274285 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2
0.4885604204274285 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2 -A
0.8168319364176728 weka.classifiers.meta.LogitBoost -P 100 -L -1.7976931348623157E308 -H 1.0 -Z 3.0 -O 1 -E 1 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.8138516935725578 weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.615748275769369 weka.classifiers.bayes.NaiveBayes 
0.5107288956916255 weka.classifiers.lazy.IBk -K 1 -W 0 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\""
0.49970796898272396 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2
0.49970796898272396 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2 -A
0.8230139745904058 weka.classifiers.meta.LogitBoost -P 100 -L -1.7976931348623157E308 -H 1.0 -Z 3.0 -O 1 -E 1 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.8154288183600676 weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.5670994617105832 weka.classifiers.bayes.NaiveBayes 
0.5137542355043367 weka.classifiers.lazy.IBk -K 1 -W 0 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\""
0.4778358145781426 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2
0.4778358145781426 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2 -A
0.819976301480968 weka.classifiers.meta.LogitBoost -P 100 -L -1.7976931348623157E308 -H 1.0 -Z 3.0 -O 1 -E 1 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.8130602712839098 weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.5545384101043519 weka.classifiers.bayes.NaiveBayes 
0.515311448273653 weka.classifiers.lazy.IBk -K 1 -W 0 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\""
0.4842705780877141 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2
0.4842705780877141 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2 -A
0.8178291102695394 weka.classifiers.meta.LogitBoost -P 100 -L -1.7976931348623157E308 -H 1.0 -Z 3.0 -O 1 -E 1 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.8143810958659647 weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.6088708005303389 weka.classifiers.bayes.NaiveBayes 
0.5086094275863173 weka.classifiers.lazy.IBk -K 1 -W 0 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\""
0.49714039109634645 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2
0.49711493803179746 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2 -A
0.8585475387913883 weka.classifiers.meta.LogitBoost -P 100 -L -1.7976931348623157E308 -H 1.0 -Z 3.0 -O 1 -E 1 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.843227271359819 weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.7829855784502836 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2 -A
0.768185478077517 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2
0.6451333150924005 weka.classifiers.bayes.NaiveBayes 
0.5145605461374861 weka.classifiers.lazy.IBk -K 1 -W 0 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\""
0.8508058650278701 weka.classifiers.meta.LogitBoost -P 100 -L -1.7976931348623157E308 -H 1.0 -Z 3.0 -O 1 -E 1 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.8419896233349973 weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.775633956694677 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2 -A
0.7618155889694169 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2
0.6470571783622083 weka.classifiers.bayes.NaiveBayes 
0.5155980678975303 weka.classifiers.lazy.IBk -K 1 -W 0 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\""
0.8520185545244311 weka.classifiers.meta.LogitBoost -P 100 -L -1.7976931348623157E308 -H 1.0 -Z 3.0 -O 1 -E 1 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.8461344017067475 weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.7818379212870403 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2 -A
0.7766065514703023 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2
0.6531404864026451 weka.classifiers.bayes.NaiveBayes 
0.5162136810587187 weka.classifiers.lazy.IBk -K 1 -W 0 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\""
0.8563516719857762 weka.classifiers.meta.LogitBoost -P 100 -L -1.7976931348623157E308 -H 1.0 -Z 3.0 -O 1 -E 1 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.8503252106090177 weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.784024362445674 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2 -A
0.7704225925807987 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2
0.6526727786599051 weka.classifiers.bayes.NaiveBayes 
0.5166074657526244 weka.classifiers.lazy.IBk -K 1 -W 0 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\""
0.8558484004527753 weka.classifiers.meta.LogitBoost -P 100 -L -1.7976931348623157E308 -H 1.0 -Z 3.0 -O 1 -E 1 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.8455943821406211 weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
0.7839413811296427 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2 -A
0.7699332085398684 weka.classifiers.trees.J48 -S -C 0.25 -B -M 2
0.6661275271306659 weka.classifiers.bayes.NaiveBayes 
0.5128497788928906 weka.classifiers.lazy.IBk -K 1 -W 0 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\""
Ensemble
	churn:     0.7203199307937453
	appetency: 0.8053084225048561
	up-sell:   0.8595914026921887
	overall:   0.79507325199693

Processing Time: PT21H2M58.460217S
End processing...
```