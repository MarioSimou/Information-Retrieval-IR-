# This file explains the content of the subfolders that are contained # in the IR main folder 

FNC_sample:
	1) Raw: 
	contains the raw FNC_1 dataset (train_bodies and train_stances) that was taken from https://github.com/FakeNewsChallenge/fnc-1/

	2) split_testing_sample.csv:
	the extracted testing subsample after the train_bodies.csv and train_stances.csv were joined and split with a ratio of 9:1. (90%)

	3) split_training_sample.csv:
	the extracted validation subsample after the train_bodies.csv and train_stances.csv were joined and split with a ration of 9:1. (10%)

img:
	1) all the images that were extracted and considered for the project

ir_package:
	1) PorterStemmer.py:
	martin porter algorithm, which used to perfom stemming on the tokens (https://tartarus.org/martin/PorterStemmer/)

	2) ProbabilisticModels.py:
	the probabilistic models that were created throught the whole project. The probabilistic models scrit implements a KL-divergence language models under the Jeliner-Mercer, Dirichlet and Bayesian smoothing. A BMS25 retrieval models is also included

	3) VSM.py:
	a vector space model (VSM) that was created for the needs of the project. The VSM model is compatible with a terf frequency weighting scheme (tfi), as well as a term frequence with inverse document term weight(tf_idf).

	4) RegressionModels.py:
	the script enables to create a Linear regression model and performs estimation based a Stochastic and Batch Gradient Descent.
	Moreover, a Logistic regression model is also created based on a Stochastic Gradient Descent.

	5) TextPreprocessing.py:
	custom script that was used to perform the steps of:
	- tokenisation
	- stop words removal
	- case folding

post_processed:
	1) processed_train_sample.csv:
	the extracted file of the processed training subsample. This training subsample after has passed through the steps of  tokenisation, stop words removal, normalisation and stemming.  
	
	2) processed_test_sample.csv:
	the extracted file of the processed testing subsample. This training subsample after has passed through the steps of  tokenisation, stop words removal, normalisation and stemming.
	
	3) body_collection.csv:
	the body collection that was processed but in that case duplicate documents werer removed. This collection was loaded in all retrieval models that were considering a background collection.
	4) scores:
		train_scores.csv :
		the training scores that were gained after the post-processed training subsample was evaluated by the retrieval models.
		Scores of BMS25, VSM (tf, tfxidf) and KL-divergence(Jeliner-Mercer,Dirichlet,Bayesian) are contained in the file

		test_scores.csv:
		the testing scores that were gained after the post-processed validation subsample was evaluated by the retrieval models. Scores of BMS25 and VSM (tfxidf) model are included, which correspond to the scores that were decided to be further examined. 

sensitivity:
	1) learning_rate_paramaters_sensitivity_analysis.csv.csv:
	contains the results of each linear/logistic model for different values of learning rate
	]2) scores_(i).csv:
	the scores that were given for all retrieval models that were evaluated under different smoothing parameters

	i -> iteration of the smoothing parameter