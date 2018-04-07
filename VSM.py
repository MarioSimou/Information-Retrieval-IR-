import re, pandas as pd, math

'''
DOCUMENTATION
------------------------------------------------------------------------------------------------------------------------
# A vector space model (VSM) is instantiated.
VSM = VSM()

# Load a dataframe as the model's collection. The name of the column that will be considered as the collection needs to 
be specified.
collection = pd.DataFrame({'Collection': ['an information retrieval model consists a mathematical model that finds values of similarities',
                                              'retrieval model is not a mathematical model','random model']})
or 
collection = pd.DataFrame({'Collection': [' '.join(['an','information', 'retrieval ','model ','consists','a','mathematical','model','that','finds','values','of','similarities']),
                               ' '.join(['retrieval', 'model', 'is', 'not', 'a' ,'mathematical' ,'model']),
                               ' '.join(['random', 'model'])]})
VSM.load_collection(collection, 'Collection')
                                             
# Calculate the cosine similarity of a query with respect to a given docuemnt. The measure/weight defines the way that 
cosine similarity is going to be calculated.
- term frequency
tf_value = VSM.cosine_similarity('retrieval model', 'an information retrieval model consists a mathematical model that finds values of similarities', measure = 'tf')
- term frequency x inverse document frequency
tf_idf_value = VSM.cosine_similarity('retrieval model', 'an information retrieval model consists a mathematical model that finds values of similarities', measure = 'tf_idf')

------------------------------------------------------------------------------------------------------------------------
'''

class VSM():

    def __init__(self, collection = {}):
        self.__collection = collection

    def cosine_similarity(self, query, document, measure ='tf_idf'):
        '''
        Cosine similarity consists a quantitative measure of similarity between a given query and document. The retrieval model
        assumes that the query and the document exist in a common vector space, having the same number of components. The query is
        tokenised in N unique terms which then, can be weighted based on two quantitative measurements:

        i) Term frequency (tfi): the number of occurences of a term in the document.
        The main drawback of term frequency is that it does not consider the discriminative power of a term. While some terms
        may occur more often than others and may be irrelevant with the query, these terms gain a high value of weight. This
        approach does not extract accurate results, and is only suggested for experimental purposes.

        ii) Inverse document Frequency x Term Frequency (tf x idf): the term frequency is normalised such as terms
        which may have a low discriminative power and are relevant to the query to gain a high value of weight. On
        the other hand, terms which have a high discriminative power are normalised such as they gain a low value of weight.

        The same process is followed for the document parameter. After the tfi or tf_t x idf_d_t is calculated, the query
        and the document are compared extract a measure of similarity. The similarity is between the interval 0 and 1. A
        value of 0 indicates no similarity, while a value of 1 denotes a high value of relevance.

        :param query: the text that is given as a query. The parameter has to be given in a string format.
        :param document: the text which a document may contain
        :param measure: defines if cosine similarity will be calculated using tfi or tf x idf

        :return:
        '''

        # Text and query parsing
        query_terms, document_terms  = re.findall('\w+',query), re.findall('\w+',document)
        # find query frequencies tfq
        query_freq = [query_terms.count(term)for term in query_terms]

        # find the query's terms that are within the document   [document_query_intersect = filter(lambda token: document_token.__contains__(token),query_token)]
        document_query_intersect = [term for term in query_terms if document_terms.__contains__(term)]

        # initialise a vector which is full of zeros -  it will be populated with the frequencies of the document's matching terms
        document_freq = [0] * len(query_freq)
        # loop over the terms of the query
        for i_term,term in enumerate(query_terms):
            # if its a term that exist in the query list
            if document_query_intersect.__contains__(term):
                # assigns the frequency in the document_freq list
                document_freq[i_term] = document_terms.count(term)

        if measure == 'tf':
            # calculates the vectors magnitude
            query_magnitude = (sum([ term**2 for term in query_freq]))**0.5
            document_mangitude = (sum([ term**2 for term in document_freq]))**0.5

            try:
                # V(a) * V(b) / |V(a)| * |V(b)| -> cosine similarity
                cosine_similarity_q_d = sum([ x_t*x for x_t,x in zip(query_freq,document_freq)])/(query_magnitude*document_mangitude)
            except ZeroDivisionError:
                cosine_similarity_q_d = 0
        elif measure == 'tf_idf':
            if self.__collection == {}:
                raise Exception('[ Tft_idft statistics cannot be calculated without importing a collection. Reinitialise the object using a collection of documents (corpus) ]')
            try:

                # tfi x idft for query and document terms
                tfi_idft_query_terms = [query_freq[i_term]*self.__calculate_idft(term) for i_term,term in enumerate(query_terms)]
                tfi_idft_document_terms = [document_freq[i_term]*self.__calculate_idft(term) for i_term,term in enumerate(query_terms)]

                # vectors mangitude
                query_magnitude = (sum([ term**2 for term in tfi_idft_query_terms]))**0.5
                document_mangitude = (sum([ term**2 for term in tfi_idft_document_terms]))**0.5

                try:
                    # V(a) * V(b) / |V(a)| * |V(b)| -> cosine similarity
                    cosine_similarity_q_d = sum([x_t * x for x_t, x in zip(tfi_idft_query_terms, tfi_idft_document_terms)]) / (query_magnitude * document_mangitude)
                except ZeroDivisionError:
                    cosine_similarity_q_d = 0


            except Exception as e:
                print('Error : {} '.format(e.__doc__))
        else:
            raise Exception('[Unknown measure option. Choose between (tf) and (tf_idf) ]')

        # checks if cosine similarity is within the range 0 and 1
#        if cosine_similarity_q_d < 0 or cosine_similarity_q_d > 1:
#            raise Exception('Cosine similarity higher of 1 and 0. Ambiguous results')
        return  cosine_similarity_q_d

    def __calculate_idft(self,term):
        collection = list(map(lambda x: list(x.values()),self.__collection['Collection'].values()))
        # frequency is set to zero
        dfreq = 0
        # number of documents of the corpus
        N = len(collection)
        # iteration over all the documents of the collection
        for document in collection:
            # if the term is contained in the document
            if document.__contains__(term):
                dfreq += 1

        # calculates the inverse document frequency
        try:
            idft = 1 + math.log10(float(N)/dfreq)
        except ZeroDivisionError:
            idft = 1

        return  idft

    def load_collection(self,collection, colname):
        '''
            A collection is loaded and converted to a data frame.

            A collection is needed so the term frequency (tf) and term frequency x inverse document frequency (tfxidf) to
            calculated. If its not, then the predictions will be biased.
        :param collection: pandas dataframe
        :param colname: column name that the method will look in the dataframe
        '''
        try:
            if isinstance(collection,object):
                collection_dictionary = {'Collection': {}}

                for document in collection[colname]:
                    document = re.findall('\w+',document)
                    document_name = ' '.join(document)
                    collection_dictionary['Collection'][document_name] = {}
                    for i_term,term in enumerate(document):
                        collection_dictionary['Collection'][document_name][str(i_term) +'_'+ term] = term
                self.__collection = collection_dictionary
        except Exception:
            print('[A collection is loaded only if its a pandas DataFrame]')


    @property
    def collection(self):
        return self.__collection

if __name__ == '__main__':
    VSM = VSM()
    collection = pd.DataFrame({'Collection': ['an information retrieval model consists a mathematical model that finds values of similarities',
                                              'retrieval model is not a mathematical model','random model']})
    '''
    # or the collection might be loaded as:
    collection = pd.DataFrame({'Collection': [' '.join(['an','information', 'retrieval ','model ','consists','a','mathematical','model','that','finds','values','of','similarities']),
                               ' '.join(['retrieval', 'model', 'is', 'not', 'a' ,'mathematical' ,'model']),
                               ' '.join(['random', 'model'])]})
    '''
    VSM.load_collection(collection, 'Collection')
    # loaded collection
    print(VSM.collection)
    # term frequency
    tf_value = VSM.cosine_similarity('retrieval model', 'an information retrieval model consists a mathematical model that finds values of similarities', measure = 'tf')
    print(f'C(query,document; term frequency) = {tf_value}')
    # term frequency x inverse document frequency
    tf_idf_value = VSM.cosine_similarity('retrieval model', 'an information retrieval model consists a mathematical model that finds values of similarities', measure = 'tf_idf')
    print(f'C(query,document; term frequency x inverse document frequency) = {tf_idf_value}')

