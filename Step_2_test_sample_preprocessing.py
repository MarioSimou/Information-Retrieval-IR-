import pandas as pd
from ir_package.ProbabilisticModels import Probabilistic_models
from ir_package.VSM import VSM
from ir_package.TextProcessing import Text_Processing
from Step_1_FNC_preprocessing_scores import scaling


if __name__ == '__main__':

    # Step 1
    # ---------------------------------------------------------------------------------------------------------------------
    # Tokenisation, Stop words removal, normalisation and stemming/lemmatisation
    # Processing of Test Sample
    test_sample = pd.read_csv('./FNC_sample/split_testing_sample.csv')
    # loads the test sample to the object                -------------------------------------------------------------
    text_collections = Text_Processing(pd.DataFrame({'Body' : test_sample.iloc[:,1],'Headline' : test_sample.iloc[:,2]}))
    # pre-processing of the collection
    text_collections.preprocessing()
    # exporting of the collection
    text_collections.export_collections(directory= './post_processed/processed_test_sample.csv')

    # Step 2
    # ---------------------------------------------------------------------------------------------------------------------
    # The validation subsample is passed through the VSM and BMS25 models. The corresponding score is calculated and stored.
    # Load Test Sample
    test_sample = pd.read_csv('./post_processed/processed_test_sample.csv', sep = ',', encoding = 'utf-8')
    # Load collection of articles bodies -> this collection does not include duplicate bodies
    body_collection = pd.read_csv('./post_processed/body_collection.csv', sep = ',', encoding= 'utf-8')

    # Creates a VSM model
    VSM = VSM()
    # load the collection's bodies
    VSM.load_collection(body_collection.iloc[:,:], 'Body')

    # Creates a probabilistic model
    probabilistic_model = Probabilistic_models()
    # loads a collection to the model
    probabilistic_model.load_collection(body_collection.iloc[:,:],'Body')

    # Initialise required list
    bms25,tfxidf= [],[]

    print('Features/Attributes values are estimated...')
    for index in range(test_sample.shape[0]): #
        if index % 100 == 0:
            print(f'Iteration: {index}')
        # Similarity results
        # VSM tfxidf
        tfxidf.append(VSM.cosine_similarity(test_sample.Headline[index], test_sample.Body[index], measure='tf_idf'))
        # BM25 model
        bms25.append(probabilistic_model.BM25(test_sample.Headline[index], test_sample.Body[index], k1 =1.2 ,k2 = 0)) # parameter k2 was identified performing a sensitivity analysis

    # scale bms25 values between 0 and 1
    s_bms25 = scaling(bms25)

    # Results Exporting
    pd.DataFrame({'BMS25' : s_bms25,'tfxidf': tfxidf}).to_csv('./post_processing/scores/test_scores.csv', encoding = 'utf-8', sep = ',',index=False)