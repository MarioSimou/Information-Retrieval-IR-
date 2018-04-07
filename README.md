### Information-Retrieval-IR-
This package corresponds to a Stance Detection project that seeked to find the relatedness between pairs of headlines and article bodies. A Vector Space Model (VSM), KL-divergence language and BMS25 model were observed to calculate the stance of a given pair. 

**VSM.py**
A vector space model (VSM) that considers a headline and body article in vector space. The VSM model is compatible with a term frequency weighting scheme (tfi), as well as a term frequence with inverse document term weight(tf_idf).

**ProbabilisticModels.py**
A probabilistic model that implements a KL-divergence language model under the Jeliner-Mercer, Dirichlet and Bayesian smoothing. A BMS25 retrieval models is also included.

**TextPreprocessing.py**
 A hardcoded script that was used to perform the steps of: 
 -tokenisation
 -stop words removal
 -case folding
 
 ----------------------------------------------------------------------------------------------------------------------------------

VECTOR SPACE MODEL IMPLEMENTATION
----------------------------------

#A vector space model (VSM) is instantiated.
>VSM = VSM()

#Load the collection   
>VSM.load_collection(collection, 'Collection')

#Calculate the cosine similarity of a query with respect to a given docuemnt. The measure/weight defines the way that  cosine similarity #is going to be calculated.
#-TERM FREQUENCY
>tf_value = VSM.cosine_similarity('retrieval model', 'an information retrieval model consists a mathematical model that finds values  of similarities', measure = 'tf')
#- term frequency x inverse document frequency
>tf_idf_value = VSM.cosine_similarity('retrieval model', 'an information retrieval model consists a mathematical model that finds values of similarities', measure = 'tf_idf')

PROBABILISTIC MODEL IMPLEMENTATION
----------------------------------
    # Create a probabilistic model
    probalistic_model = Probabilistic_models()

    # Create a collection
    collection = pd.DataFrame({'Collection': ['an information retrieval model consists a mathematical model that finds values of similarities',
                                                  'retrieval model is not a mathematical model','random model']})
    or the collection might be similarly loaded as:
    collection = pd.DataFrame({'Collection': [' '.join(['an','information', 'retrieval ','model ','consists','a','mathematical','model','that','finds','values','of','similarities']),
                                   ' '.join(['retrieval', 'model', 'is', 'not', 'a' ,'mathematical' ,'model']),
                                   ' '.join(['random', 'model'])]})
    # Load the collection
    probalistic_model.load_collection(collection, 'Collection')

    Choose the right smoothing technique
    
    # LINEAR INTERPOLATION
    # Jeliner-Mercer Smoothing
    # specifying a method of 'jeliner-mercer', a jeliner-mercer smoothing is performed
    # - the parameter of jeliner-mercer tecnique is checked to be within the range of 0 and 1
    
    jeliner_value = probalistic_model.KL_divergence('retrieval model', 'an information retrieval model consists a mathematical model                            that finds values of similarities', method = 'jeliner-mercer',log= True, parameter= 0.5)
    
    # Dirichlet
    # specifying a method of 'dirichlet', a dirichlet smoothing is performed
    
    dirichlet_value = probalistic_model.KL_divergence('retrieval model', 'an information retrieval model consists a mathematical model                          that finds values of similarities', method = 'dirichlet',log= True, parameter= 5)
    # BAYESIAN SMOOTHING
    # Bayesian
    # specifying a method of 'bayesian', a bayesian smoothing is performed

    bayesian_value = probalistic_model.KL_divergence('retrieval model','an information retrieval model consists a mathematical model                            that finds values of similarities', method='bayesian', log=True, parameter=0.5)

    # BMS25 Retrieval model    
    # BMS25
    # the parameters k1 and k2 needs to be defined
    
    bms25_value = probalistic_model.BM25('retrieval model','an information retrieval model consists a mathematical model that finds                           values of similarities',k1 =1.2 ,k2 =5)
    
