from string import punctuation
from ir_package.PorterStemmer import PorterStemmer
import re, pandas as pd

class Text_Processing():

    def __init__(self, train_sample):
        try:
            # initialise the extend of the data.frame
            self.__n,self.__m = train_sample.shape[0],train_sample.shape[1]
            # initialise the columns names
            self.__columns_names = (list(train_sample.columns))
            # assigns the collection
            self.__collections = train_sample
            self.__number_list = None

        except Exception as e:
            if not type(train_sample) == 'DataFrame':
                print('[ Unknown Data Structure. Set a data frame structure ]')
            else:
                print('[ Unknown Error ]')

    def __stop_words_removal_and_stemming(self):
        try:
            # initialise the stemming class (Available Code: https://tartarus.org/martin/PorterStemmer/def.txt)
            stemming = PorterStemmer()
            # set the collection's dictionary as variable
            collections = self.__collections
            # create a new list which will contain the updated tokens
            collections_up = {}
            # loop over the items of the dictionary
            for i_collection,collection in enumerate(collections):
                collections_up[collection] = {}
                # stop words are calculating for the current item
                stop = [word for freq, word in self.__get_stop_words()[i_collection]]
                # loop over the titles of the dictionary
                for document in collections[collection]:
                    collections_up[collection][document] = {}
                    # loop over the tokens of each title
                    for token in collections[collection][document].values():
                        # stop words are compared with the token
                        if not stop.__contains__(token):
                            # if the token is not contained in the stop words list, then it passes to the dictionary
                            # before the token is passed to the dictionary, its also stemmed
                            collections_up[collection][document][token] = stemming.stem(token,0,(len(token)-1))
            # collection i updated
            self.__collections = collections_up
        except Exception as e:
            print('[ Error while stop words were removing.\nException: {} ]'.format(e.__doc__))

    def __get_stop_words(self, n_stop = 8 ):

            # initialise a tokens list
            tokens = []
            # set distribution of the tokens
            #self.__set_concat_tokens()

            # iterate over the colection's tokens, finds the corresponding frequenecy and store the results in the tokens list
            for i_item,item in enumerate(self.__get_concat_tokens().values()):
                tokens.append([])
                for i_token,token in enumerate(set(item)):
                    freq= 1         # initial frequency for each word
                    index = i_token +1      # initialise the index which the token will start counting similar words
                    while(index < len(item) ):
                        if token == item[index]:    # if the token is equal with a token that follows, then the frequency is increased
                            freq += 1   # frequency
                        index +=1       # index

                    # the frequency and the token are passsed in the list as a tuple
                    tokens[i_item].append((freq,token))

            # tokens are sorted by their frequency
            sorted_tokens = [sorted(item, key = lambda frequency: frequency[0]) for item in tokens]
            # sorted tokens are sorted based on their rank
            #freq_tokens = [[token for token in item if (item.index(token) < (n_stop/2.0)) or (item.index(token) >= (len(item) -(n_stop/2.0)))] for item in sorted_tokens]
            stop_words= [list(filter(lambda token: (item.index(token) < (n_stop/2.0)) or (item.index(token) >= (len(item)-(n_stop/2.0))),item)) for item in sorted_tokens]

            # frequently tokens are assign to the object
            return  stop_words


    def __tokenization(self):
        #try:
            # sets the puntuation terms
            punctuation_pattern= '[' + punctuation + "’'—â€™˜“”]+"
            # initialise a list that will include the tokenized text
            number_l = {'Number_list' :  []}
            collections = {}
            for i_collection,collection in enumerate(self.__columns_names): # loop over the columns of the train_sample
                collections[collection] = {}
                # Initialise a dictionary that will pass into the list
                for i_document,document in enumerate(self.__collections[collection]):                      # loop over the records of the columns
                    if i_document % 10000 == 0:
                        print('Processing record {} of {} of column collumn {} ...'.format(i_document+1,self.__n,collection))

                    #document_name = document
                    document_name = '_'.join([str(i_document),document])
                    collections[collection][document_name] = {}
                    # checks if the context of the training sample contains any whitespace character and removes it (scrabing using regular expression)
                    try:
                        doc_words = re.findall(r'\S+', document)
                    except TypeError:
                        print('[Unrecognized numerical value while it was expecting string.. Casting is performed on the column {}. The values will pass to the numerical list... ]'.format(self.__columns_names[i_collection]))
                        doc_words = re.findall(r'\S+', str(document))
                    for term in doc_words:              # loop over the terms and marks them if they contain punctuation or not
                        # Case-Folding / all terms are converted to lowercase
                        term = term.lower()         # all terms ar lower case, similarly as the vocabulary #  normalisation ()
                        # loop over all the characters withing the variable punctuation terms
                        if re.search(punctuation_pattern, term):
                            match = re.search(punctuation_pattern, term)
                            if match.start() == 0 or match.end() == (len(term)):
                                term_up = re.findall('\w+', term)
                                if len(term_up) == 1 and term_up[0] is not '':
                                    collections[collection][document_name][term] = term_up[0]
                                elif len(term_up) == 2:
                                    if term.__contains__("’") or term.__contains__("'"):
                                        collections[collection][document_name][term] = "'".join(term_up)
                                    elif term.__contains__('-'):
                                        collections[collection][document_name][term] = "-".join(term_up)
                                    elif term.__contains__('.'):
                                        if bool(re.search('[1-9]+',term)):
                                            number_l['Number_list'].append(term_up)
                                        else:
                                            term_up = re.findall(r'\w+',term)
                                            if len(term_up) > 1:
                                                collections[collection][document_name][term] = term_up[0] + '.' + term_up[1]
                                    else:
                                        continue
                        else:
                            if re.search('[1-9]+',term):
                                number_l['Number_list'].append(term)
                            else:
                                collections[collection][document_name][term] = term
            # reassigning object variables
            self.__collections = collections
            self.__number_list = number_l
        #except Exception as e:
            #print('[ Error during the tokenization : {} ]'.format(e.__doc__))

    def __get_concat_tokens(self):
        # get the whole distribution of each column, without considering their relative position
        try:
            d = {}
            for i in self.collections:
                d[i] = []
                for j in self.collections[i]:
                    for k in self.collections[i][j].values():
                        d[i].append(k)
            # set concatenated tokens (concatenated means the whole distribution of words)
            return d
        except Exception as e: \
                print('[ Error while the tokens where concatenating\nExceptio : {}]'.format(e.__doc__))


    def export_collections(self, directory = './', sep =',', encoding = 'utf-8', index = True):
        try:
            print('[ Exporting process has started... Data are processing... ]')
            # raw collection
            collections = self.__collections
            # collections' names
            collections_names = [collection for collection in collections]

            # converts the collection in a nested list
            processed_text = [pd.DataFrame({collections_names[i_c]:c}) for i_c,c in enumerate([[[token for token in collections[collection][document].values()] for document in collections[collection]] for collection in collections])]
            # Exporting the list
            pd.concat(processed_text,axis = 1).to_csv(directory,encoding= 'utf-8', index = True, index_label='id')

        except Exception as e: \
                print('[ Error during the export : {} ]'.format(e.__doc__))


    def preprocessing(self):
        # tokenise the given text
        self.__tokenization()
        # remove the stop words and perform stemming
        self.__stop_words_removal_and_stemming()

    @property
    def collections(self):
        return self.__collections
    @property
    def colnames(self):
        return  self.__columns_names

