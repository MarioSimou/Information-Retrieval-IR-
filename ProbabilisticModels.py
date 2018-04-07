import re,math,pandas as pd

'''
DOCUMENTATION
------------------------------------------------------------------------------------------------------------------------

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
    
    jeliner_value = probalistic_model.KL_divergence('retrieval model', 'an information retrieval model consists a mathematical model that finds values of similarities', method = 'jeliner-mercer',log= True, parameter= 0.5)
    
    # Dirichlet
    # specifying a method of 'dirichlet', a dirichlet smoothing is performed
    
    dirichlet_value = probalistic_model.KL_divergence('retrieval model', 'an information retrieval model consists a mathematical model that finds values of similarities', method = 'dirichlet',log= True, parameter= 5)
# BAYESIAN SMOOTHING
    # Bayesian
    # specifying a method of 'bayesian', a bayesian smoothing is performed

    bayesian_value = probalistic_model.KL_divergence('retrieval model','an information retrieval model consists a mathematical model that finds values of similarities', method='bayesian', log=True, parameter=0.5)

# BMS25 Retrieval model    
    # BMS25
    # the parameters k1 and k2 needs to be defined
    
    bms25_value = probalistic_model.BM25('retrieval model','an information retrieval model consists a mathematical model that finds values of similarities',k1 =1.2 ,k2 =5)
    
------------------------------------------------------------------------------------------------------------------------
'''

class Probabilistic_models():
    def __init__(self, collection = {}):
        self.__collection = collection

    def load_collection(self,collection, colname):
        '''
            A collection is loaded and converted a pandas data frame in a dictionary
        :param collection: pandas data frame
        :param colname: column name that the method will look in the data frame
        '''
        try:
            if isinstance(collection,object):
                collection_dictionary = {'Collection': {}}
                for document in collection[colname]:
                    document = re.findall('\w+',document)
                    document_name = ' '.join(document)
                    collection_dictionary['Collection'][document_name] = {}
                    for i_term,term in enumerate(document):
                        collection_dictionary['Collection'][document_name][str(i_term) +'_'+term] = term
                self.__collection = collection_dictionary

                # Creates the correspond language model for the collection
                print('Collection is loaded... The corresponded Language model is also calculated..')
                self.__construct_collection_model()

        except Exception:
            print('[A collection is loaded only if its a pandas DataFrame]')

    def __construct_collection_model(self):
            if self.__collection == {}:
                raise Exception('[Load a collection before any probabilistic calculation]')
            vc = [element for inner_list in [list(term.values()) for document in self.__collection.values() for term in document.values()] for element in inner_list]
            # Create the Collection Model Mc and assign the unique terms
            model= (list(map(lambda term: vc.count(term) / float(len(vc)), set(vc))),list(set(vc)))

            self.__collection_list = vc
            self.__collection_model = model

    def BM25(self,query,document,k1,k2):
        '''

        :param query:
        :param document:
        :param k1:
        :param k2:
        :return:
        '''

        # initialise the collection
        collection = self.__collection
        # construct the query's and document's terms
        query_terms, document_terms = re.findall('\w+', query), re.findall('\w+', document)

        # Initial variables
        # Length of the collection
        N = len(collection['Collection'])
        # length of the document
        d = len(document_terms)
        # average length of the collection
        md = len(self.__collection_list) / len(collection['Collection'])
        b = 0.75
        k = k1 * ((1 - b) + b * (d / float(md)))
        # need to be set correctly
        s, S = 0, 0
        collection_document_list = [[list(term.values()) for term in document.values()] for document in collection.values()][0]
        eBM25 = 0

        for term in query_terms:
            # query frequency of term i
            qfi = query_terms.count(term)
            # term frequency for the document
            tfi = document_terms.count(term)
            # document frequency of a term
            dfi = len(list(filter(lambda x: x.__contains__(term),collection_document_list)))
            # estimated BM25
            eBM25 += math.log(((s+0.5)/(S-s+0.5))/((dfi-s+0.5)/(N-dfi-S+s+0.5)))*(((k1+1)*tfi)/(k+tfi))*(((k2+1)*qfi)/(k2+qfi))

        return eBM25


    def KL_divergence(self,query,document, method ='jeliner-mercer', parameter = 0.5 , log = True):

        '''
        Kullback-Leibler divergence corresponds to a language model approach which compares a Query (Mq) with a Document (Md)
        language model. This approach constitutes the comparison between a Query's and Document's probability
        distribution and attempts to estimates the similarity/entropy between the two distributions. In that case,
        the probability distribution is estimated using the relative frequency of each individual term. Therefore, the
        relative frequency derives dividing the frequency of a term i in the document unit with the document's length.
        Estimating the probability using the relative frequency, the expected values are the most likely that we can estimate.
        For that reason the estimated KL divergence model is based on a Maximum Likelihood Estimation (MLE).

        The estimated divergence is then estimated according the following equations:

            R (document,query) = KL (Md || Mq) = Π P(ti|Mq) * [ P(ti|Mq) / P(ti|Md)] , for i = t ε q [eq. 1]

            The model is decomposed in a summation estimating the log of likelihood ratio:

            R (document,query) = KL (Md || Mq) = Σ P(ti|Mq) * log [ P(ti|Mq) / P(ti|Md)] , for i = t ε q [eq. 2]

        P(ti|Mq) = the probability of a term i in respect to a query language model
        P(ti|Md) = the probability of a term i in respect to a document language model
        P(ti|Mq) / P(ti|Md) = likelihood ratio between the two language models
        # Mq distribution is the target distribution which we approximate using the Md

        t ε q =  terms which are within the query

        Based on the unigram assumption, the probability of each individual term is estimated independently. This means
        that the query can be vectorised and each term to provide an estimate. Additionally, the product of all estimates,
        which derive from a query, provides the probability of the query with respect to a language model [P(ti|Md) or P(ti|Mc)].

        Therefore, calculating the KL estimate for each individual term, an estimate is extracted, which infers how similar
        the distributions of a Query with a Document model (look on the likelihood ratio) are [eq. 1].
        The same model is decomposed in a summation calculating the logarithm of likelihood ratio [eq. 2].

        Smoothing techniques are also applicable for the KL divergence model. A linear interpolation (Jelinek_Mercer or
        Dirchlet) and a Bayesian smoothing are permitted using this package. Smoothing techniques attempt
        to perform a discount in non zero probabilities and they also attempt to give some probability to unseen words.
        For that reason, the probability of a term in respect to a document does not depend only on the document's
        distribution. The concept of collection's distribution is integrated.

        1) Linear Interpolation:
            i) Jelinek - Mercer:
                    P(ti|Md) = λ * P(ti|Md) + (1-λ) * P(ti|Mc)

                    λ = smoothing parameter which assigns a weight in P(ti|Md) and P(ti|Mc).
                Condition:
                    0 < λ < 1
            ii) Dirichlet:
                    P(ti|Md) = |d|/(|d|+m) * P(ti|Md) + m/(|d| + m) * P(ti|Mc)

                    |d| = number of word occurences in the document
                    m = average length of a document

        2) Bayesian:
                    P(ti|d) = tfi + a * P(ti|Mc) / (|d| + a)

                    tfi = term frequency of term i in the document
                    a = Bayesian prior. Indicates the strength of our belief in uniformity
                    |d| = document length


        Results:
            A value close to 0 indicates that Mq and Md are very similar. A value higher of 0 indicates that the
            language model are disimilar.

        Parameters Setting:
        A small value of λ or a large value of α typically assign more weight in the collections language model(Mc) and
        vice versa.
        Default:
            λ = 0.5 -> Md and Mc models contribute the same for the P(ti|Md) estimate

        :param query: a string or a list of words which represent the query's terms
        :param document: a string or a list of words which represent the document's terms
        :param method : set the method for smoothing. Jelinek - Mercer, Dirichlet  and Bayesian smoothing are enabled
        to be used
        :param parameter = set the smoothing techniques' parameter. The default method is Jelinek - Mercer and a
        parameter of 0.5 is used.
                Jelinek - Mercer: the smoothing parameter must be within the interval of 0 and 1. Is specified by a
                boolean value.

                Dirichlet: A constant value

                Bayesian: the smoothing parameter may be a boolean or int value. No limit for the specified values.

        :param log: set if the KL-divergence will be estimated as a product or a logarithm summation
        :return:
        '''


        # Text and query parsing
        # Set the collection variable
        collection = self.__collection

        # construct the query's and document's terms (vectors)
        vq, vd = re.findall('\w+', query), re.findall('\w+', document)

        # Calculates the probabilities for collection's, document's and query's terms
        # set language models list
        l = [vq,vd]
        Mq,Md= [(list(map(lambda term: collection.count(term)/float(len(collection)),set(collection))),list(set(collection))) for collection in l]

        # unpack language models and unique vectors
        Mq,un_vq = Mq    # Query Model, Unique Vector of query terms
        Md,un_vd = Md    # Document Model, Unique Vector of document terms
        Mc,un_vc = self.__collection_model  # Collection Model, Unique Vector of collection terms

        # Check condition -> sum(Mq) = 1
        # Condition
        for probabilities in [Mq,Md,Mc]:
            # Calculate the sum of probabilities
            sum_prob = sum(probabilities)
            # look in the interval  0.99999 < x < 1.00001
            if (sum_prob) < 0.99999 or (sum_prob) > 1.00001:
                raise Exception('[ The sum of probabilities is not equal of 1. Biased Language models]')


        # language models are calculated using KL divergence algorithm
        # Initialise the Kullback-leibler variable
        eKL_Md_Mq  = 0


        # Assign the corresponding probability of each query term in respect to Mq,Md,Mc
        # loop over the query terms
        for i_q_term,q_term in enumerate(un_vq):
            # Calculates P(ti|Mq)
            P_t_Mq = Mq[i_q_term]

            # Calculates P(ti|Md) and P(ti|Mc)
            # if the collection vector contains the query term
            if un_vc.__contains__(q_term):
                # assign the correspond collection's probability of the query term
                P_t_Mc = Mc[un_vc.index(q_term)]
                # checks if the document vector contains the query term
                if un_vd.__contains__(q_term):
                    # if true, the corresponding probability value is assigned
                    P_t_Md = Md[un_vd.index(q_term)]
                else:
                    # if not, a value of zero is assigned
                    P_t_Md = 0
            else:
                # if the query term is not contained in the collection, P_t_Mc and P_t_Md are zero
                P_t_Mc = 0
                P_t_Md = 0


            # Parameters which may needed
            nd = len(vd)
            nq = len((vq))

            # Smoothing
            if method == 'jeliner-mercer':
                if parameter >= 0 and parameter <= 1:
                    # Smoothed probability
                    P_t_Md_smoothed = (parameter * P_t_Md + (1 - parameter) * P_t_Mc)
                else:
                    raise Exception('[For jeliner-mercer smoothing the value of λ must in the interval of 0 and 1]')
            elif method == 'dirichlet':
                    # Smoothed Probability
                    P_t_Md_smoothed = (nd / (nd + parameter)) * P_t_Md + (parameter / (parameter + nd)) * P_t_Mc
            elif method == 'bayesian':
                # Bayesian Smoothing
                P_t_Md_smoothed = (vq.count(q_term) + parameter * P_t_Mc) / (nd + parameter)
            else:
                raise Exception("[Unrecognized smoothing technique. Choose a technique between ['jeliner-mercer','dirchlet','bayesian']")

            # Kullback-Leibler entropy - measure of similarity between the language models
            # checks the divergence of the Mq in respect to Md (combined probability of Mq and Mc)
            try:
                if log:
                    # eq. 2
                    eKL_Md_Mq += P_t_Mq * math.log10(P_t_Mq /P_t_Md_smoothed)
                elif not log:
                    # eq. 1
                    eKL_Md_Mq *= P_t_Mq * (P_t_Mq / P_t_Md_smoothed)
                else:
                    raise Exception('[Unrecognised log parameter]')
            except ZeroDivisionError:
                # zero division error
                #print('Zero division')
                eKL_Md_Mq += 0     #1.0 / nq
        # Condition
#        if eKL_Md_Mq >= 0:
        return eKL_Md_Mq
#        else:
#            raise Exception('Negative Value. Incorrect calculations')
    @property
    def collection(self):
        return  self.__collection

if __name__ == '__main__':

    probalistic_model = Probabilistic_models()
    collection = pd.DataFrame({'Collection': ['an information retrieval model consists a mathematical model that finds values of similarities',
                                              'retrieval model is not a mathematical model','random model']})
    '''
    # or the collection might be loaded as:
    collection = pd.DataFrame({'Collection': [' '.join(['an','information', 'retrieval ','model ','consists','a','mathematical','model','that','finds','values','of','similarities']),
                               ' '.join(['retrieval', 'model', 'is', 'not', 'a' ,'mathematical' ,'model']),
                               ' '.join(['random', 'model'])]})
    '''
    probalistic_model.load_collection(collection, 'Collection')
    # loaded collection
    print(probalistic_model.collection)

    # LINEAR INTERPOLATION
    # Jeliner-Mercer Smoothing

    jeliner_value = probalistic_model.KL_divergence('retrieval model', 'an information retrieval model consists a mathematical model that finds values of similarities', method = 'jeliner-mercer',log= True, parameter= 0.5)
    print(f'D_jeliner(query,document) = {jeliner_value}')
    # Dirichlet
    dirichlet_value = probalistic_model.KL_divergence('retrieval model', 'an information retrieval model consists a mathematical model that finds values of similarities', method = 'dirichlet',log= True, parameter= 5)
    print(f'D_dirichlet(query,document) = {dirichlet_value}')
    # Bayesian
    bayesian_value = probalistic_model.KL_divergence('retrieval model','an information retrieval model consists a mathematical model that finds values of similarities', method='bayesian', log=True, parameter=0.5)
    print(f'D_bayesian(query,document) = {bayesian_value}')

    # Probabilistic Model
    # BMS25
    bms25_value = probalistic_model.BM25('retrieval model','an information retrieval model consists a mathematical model that finds values of similarities',k1 =1.2 ,k2 =5)
    print(f'D_bms25(query,document) = {bms25_value}')

