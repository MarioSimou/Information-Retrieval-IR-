### Information-Retrieval-IR-
This package corresponds to a Stance Detection project that seeked to find the relatedness between pairs of headlines and article bodies. A Vector Space Model (VSM), KL-divergence language and BMS25 model were observed to calculate the stance of a given pair. [More details](https://drive.google.com/open?id=18MBpPbfj1KRMH28Sd-I8bScS3d4q2OFw)

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

__A vector space model (VSM) is instantiated__
```
VSM = VSM()
```

__Load the collection__   
```
VSM.load_collection(collection, 'Collection')
```

__Calculate the cosine similarity of a query with respect to a given docuemnt. The measure/weight defines the way that  cosine similarity #is going to be calculated__

**TERM FREQUENCY**
```
tf_value = VSM.cosine_similarity('retrieval model', 'an information retrieval model consists a mathematical model that finds values  of similarities', measure = 'tf')
```

**TERM FREQUENCY x INVERSE DOCUMENT FREQUENCY**
```
tf_idf_value = VSM.cosine_similarity('retrieval model', 'an information retrieval model consists a mathematical model that finds values of similarities', measure = 'tf_idf')
```

PROBABILISTIC MODEL IMPLEMENTATION
----------------------------------

__Create a probabilistic model__
```
probalistic_model = Probabilistic_models()
```

__Load the collection__
```
probalistic_model.load_collection(collection, 'Collection')
```

**Smoothing Techinques**

__LINEAR INTERPOLATION__
1. Jeliner-Mercer Smoothing
```
jeliner_value = probalistic_model.KL_divergence('retrieval model', 'an information retrieval model consists a mathematical model that finds values of similarities', method = 'jeliner-mercer',log= True, parameter= 0.5)
```    
2. Dirichlet
```
dirichlet_value = probalistic_model.KL_divergence('retrieval model', 'an information retrieval model consists a mathematical model that finds values of similarities', method = 'dirichlet',log= True, parameter= 5)
```
__BAYESIAN SMOOTHING__
1. Bayesian
```
bayesian_value = probalistic_model.KL_divergence('retrieval model','an information retrieval model consists a mathematical model that finds values of similarities', method='bayesian', log=True, parameter=0.5)
```

BMS25 Retrieval model    
----------------------
__BMS25__
```
bms25_value = probalistic_model.BM25('retrieval model','an information retrieval model consists a mathematical model that finds values of similarities',k1 =1.2 ,k2 =5)
``` 
