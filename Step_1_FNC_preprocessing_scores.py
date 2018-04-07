import pandas as pd
from ir_package.VSM import VSM
from ir_package.TextProcessing import Text_Processing
from ir_package.ProbabilisticModels import Probabilistic_models
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
from scipy.stats import boxcox,kstest
from scipy.stats.mstats import  normaltest

style.use('ggplot')

# Custom Functions
# ----------------------------------------------------------------------------------------------------------------------
# scales an input list at the range of 0-1
def scaling(unscaled_values):
    min_v = min(unscaled_values)
    max_v = max(unscaled_values)
    return list(map(lambda x: (float(x)-min_v)/(max_v-min_v),unscaled_values))


# KL divergence function
def KL_divergence(p,q):
    # Equation : KL(p,q) = Sum (p(x) * log(p(x)/q(x)))
    a = np.asarray(p, dtype=np.float)
    b = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

# Compares P(X) with respect Q(X), a distance divergence is identified
def scores_comparison(df,names,width = 0.25, n_plot = (1,2)):
    x_ref = np.asarray([0.875, 0.625, 0.375, 0.125])
    y_ref_prob = [0.0736,0.1783,0.0168,0.74]

    i,j = 0,0

    f, axarr = plt.subplots(n_plot[0],n_plot[1], sharey=True,sharex=True)
    for name in names:
        if j == n_plot[1]:
            i+=1
            j = 0

        y = np.asarray(sorted(df[name].tolist()))
        # length of y
        ny = len(y)
        # estimates the corresponding probabilities
        y_prob = [len(y[(y > (value - width/2)) & (y < (value + width/2))])/float(ny) for value in x_ref]
        y_prob = np.asarray(y_prob)
        print(f'{name} - KL Divergence Value : {KL_divergence(y_ref_prob,y_prob)}')

        try:
            axarr[i, j].bar(x_ref,y_ref_prob, width, fill = False, edgecolor = ['red','red','red','red'], label = '$P_{ref}(X)$')
            axarr[i, j].bar(x_ref,y_prob, width ,color = 'green', alpha = 0.5, label = '$Q_{'+ name+'}(X)$')
            axarr[i, j].set_ylabel('$\Pr(Stance_i)$')
            axarr[i, j].set_xlabel('$Stance_i$')
            axarr[i, j].set_xticks(x_ref, ['Agree', 'Discuss', 'Disagree', 'Unrelated'])
            axarr[i, j].text(x_ref[1],y_ref_prob[3]-0.2, '$D(P|Q) = '+str(round(KL_divergence(y_ref_prob,y_prob),2)) + '$')
            axarr[i, j].legend(loc = 1)
        except:
            axarr[j].bar(x_ref, y_ref_prob, width, fill=False, edgecolor=['red', 'red', 'red', 'red'], label='$P_{ref}(X)$')
            axarr[j].bar(x_ref, y_prob, width, color='green', alpha=0.5, label=name)
            axarr[j].set_title(name + '$\/vs\/Reference\/Histogram$')
            axarr[j].set_xticks(x_ref, ['Agree', 'Discuss', 'Disagree', 'Unrelated'])
            axarr[j].set_ylabel('$P(Stance_i)$')
            axarr[j].set_xlabel('$Stance_i$')
            axarr[j].legend(loc=1)
        j+=1

    # Fine-tune figure; make subplots farther from each other.
    f.subplots_adjust(hspace=0.3)
    plt.savefig('./img/distances_mertic.png')
    plt.show()


# ---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Step 1
    # ------------------------------------------------------------------------------------------------------------------
    # Files loading
    # Raw Datasets
    bodies = pd.read_csv('./FNC_sample/Raw/train_bodies.csv')   # 28 items are similar
    headings = pd.read_csv('./FNC_sample/Raw/train_stances.csv')

    # Calculating the Percentages
    # Files Merging
    df = pd.merge(left = headings, right = bodies, left_on= str(headings.columns[1]), right_on= str(bodies.columns[0]))
    # creates a data frame which contains only the Headline and Article Body
    df_new = pd.DataFrame({'Headline' : df['Headline'], 'Body': df['articleBody']})

    # Stances
    results = list(df.Stance.values)
    # Number of unrelated stances
    n_unrelated = results.count('unrelated') # 36545
    # Number of discuss stances
    n_discuss = results.count('discuss') # 8909
    # Number of agree stances
    n_agree = results.count('agree') # 3678
    # Number of disagree stances
    n_disagree = results.count('disagree') # 840
    # Total Number
    n = sum([n_unrelated,n_discuss,n_agree,n_disagree])
    print('Unrelated: {}\nDiscuss: {}\nAgree: {}\nDisagree: {}'.format(n_unrelated,n_discuss,n_agree,n_disagree))
    # checks if there are any null values
    print('Contain null values:\n\tHeadline: {}\n\tBody: {}\n'.format(any(list(df_new.isnull().iloc[:,0])),any(list(df_new.isnull().iloc[:,1]))))

    # Train/Test Split
    merge= pd.DataFrame({'Headline': df.Headline,'Body':df.articleBody,'Stance': df.Stance})
    # group the data frame based on the Stance option
    groups = list(merge.groupby('Stance'))

    # Agree Dataframe
    agree = groups[0][1]
    # Disagree Df
    disagree = groups[1][1]
    # Discuss Df
    discuss = groups[2][1]
    # Unrelated Df
    unrelated = groups[3][1]

    # initialise the train and test sample
    train_sample,test_sample  = [],[]

    check = [n_agree,n_discuss,n_disagree,n_unrelated]
    for i_item,item in enumerate([agree,discuss,disagree,unrelated]):
        X_train, X_test, Y_train, Y_test = train_test_split(item.iloc[:,0:2], item.iloc[:,2], test_size=0.1)  # explanatory, response, percentage of testing sample

        # Display
        name = str(groups[i_item][0]).upper()
        X_train_p = X_train.shape[0]/n*100
        X_test_p = X_test.shape[0]/n*100
        print('-----------------------------------------------------------------------------------------------------------')
        print(f'{name}\nTrain: {X_train_p:.2f}%\tTest: {X_test_p:.2f}%')
        print(f'Total: {X_train_p+X_test_p:.2f}%')
        print(f'Check Equality: {X_train_p+X_test_p:.2f}% almost {check[i_item]/n*100:.2f}%'.format(X_train_p+X_test_p,check[i_item]/n*100))

        # Add the samples in a list
        train_sample.append(pd.concat([X_train,Y_train],axis = 1, join = 'inner'))
        test_sample.append(pd.concat([X_test,Y_test],axis= 1, join= 'inner'))
    print('---------------------------------------------------------------------------------------------------------------')
    print(f'Total Percentage Train: {sum([len(item) for item in train_sample])/n*100:.2f}%')
    print(f'Total Percentage Train: {sum([len(item) for item in test_sample])/n*100:.2f}%')

    # Raw - Test/Train Split Comparison
    raw= [value/float(n)*100 for value in (n_agree,n_discuss,n_disagree,n_unrelated)]
    test = [value/float(n)*100 for value in (test_sample[0].shape[0],test_sample[1].shape[0],test_sample[2].shape[0],test_sample[3].shape[0])]
    train = [value/float(n)*100 for value in (train_sample[0].shape[0],train_sample[1].shape[0],train_sample[2].shape[0],train_sample[3].shape[0])]
    labels = [1,2,3,4]
    labels_x = ['Agree','Discuss','Disagree','Unrelated']


    # plot
    barWidth = 0.85
    # Initialise a figure
    plt.figure(1)
    # sets a subplot
    plt.subplot(211)
    # sets a title
    plt.title(r'$\mathbf{Raw - Train\//\/Test\/\/Split\/\/Comparison}$')
    # Create raw plot
    plt.bar(labels, raw, color='#f9bc86', edgecolor='white', width=barWidth)
    plt.text(labels[0]-.15, raw[0]+1, r'$\mathit{'+ str(round(raw[0],2)) + '}\/\%$', {'color': '#ff8533', 'fontsize': 10})
    plt.text(labels[1]-.15, raw[1]+1, r'$'+str(round(raw[1],2)) + '\/\%$',{'color': '#ff8533', 'fontsize': 10})
    plt.text(labels[2]-.15, raw[2]+1, r'$'+str(round(raw[2],2)) + '\/\%$',{'color': '#ff8533', 'fontsize': 10})
    plt.text(labels[3]-.15, raw[3]+1, r'$'+str(round(raw[3],2)) + '\/\%$',{'color': '#ff8533', 'fontsize': 10})
    plt.xticks(labels,labels_x)

    # Custom y axis
    plt.ylabel(r'$\mathit{Percentage\/\%}$')
    plt.subplot(212)
    # Create train plot
    plt.bar(labels, train, color='#f9bc86', edgecolor='white', width=barWidth, label = 'Train')
    plt.text(labels[0]-.15, 0, r'$\mathit{'+ str(round(train[0],2)) + '}\/\%$', {'color': '#BA7B44', 'fontsize': 10})
    plt.text(labels[1]-.15, train[1]/3.0, r'$\mathit{'+ str(round(train[1],2)) + '}\/\%$',{'color': '#BA7B44', 'fontsize': 10})
    plt.text(labels[2]-.15, 1, r'$\mathit{'+ str(round(train[2],2)) + '}\/\%$',{'color': '#BA7B44', 'fontsize': 10})
    plt.text(labels[3]-.15, train[3]/2, r'$\mathit{'+ str(round(train[3],2)) + '}\/\%$',{'color': '#BA7B44', 'fontsize': 10})

    # Create test plot
    plt.bar(labels, test, bottom= train, color='#a3acff', edgecolor='white', width=barWidth,label = 'Test' )
    plt.text(labels[0]-.15, train[0]+2, r'$\mathit{'+ str(round(test[0],2)) + '}\/\%$', {'color': '#303668', 'fontsize': 10})
    plt.text(labels[1]-.15, train[1]+2, r'$\mathit{'+ str(round(test[1],2)) + '}\/\%$',{'color': '#303668', 'fontsize': 10})
    plt.text(labels[2]-.15, train[2]+5, r'$\mathit{'+ str(round(test[2],2)) + '}\/\%$',{'color': '#303668', 'fontsize': 10})
    plt.text(labels[3]-.15, train[3]+1, r'$\mathit{'+ str(round(test[3],2)) + '}\/\%$',{'color': '#303668', 'fontsize': 10})

    plt.legend(loc = 'upper left')
    # Custom x axis
    plt.xticks(labels,labels_x)
    # Custom y axis
    plt.ylabel(r'$\mathit{Percentage\/\%}$')
    # save figure
    plt.savefig('./img/test_train_split.png')
    # Show graphic
    plt.show()

    # Training Sample Export
    pd.concat(train_sample, axis = 0).to_csv('./FNC_sample/split_training_sample.csv',encoding='utf-8', index= True, index_label='id')
    # Testing Sample Export
    pd.concat(test_sample, axis = 0).to_csv('./FNC_sample/split_testing_sample.csv',encoding='utf-8', index= True, index_label='id')

    # Step 2
    #-----------------------------------------------------------------------------------------------------------------------

    # Test Sample Load
    training_sample_raw = pd.read_csv('./FNC_sample/split_training_sample.csv', encoding='utf-8')

    # Text Pre-processing - Data Cleaning
    # loads the test sample to the object
    text_collections = Text_Processing(pd.DataFrame({'Body' : training_sample_raw.iloc[:,1],'Headline' : training_sample_raw.iloc[:,2]}))
    # pre-processing of the collection
    text_collections.preprocessing()
    # exporting of the collection
    text_collections.export_collections(directory= './post_processed/processed_train_sample.csv')

    # NOTE: the same pre-processing task was performed for the again for the body collection such as duplicate article bodies to be removed

    # Step 3
    #-----------------------------------------------------------------------------------------------------------------------
    # Implementing VSM, BMS25 and Kl-divergence models, the corresponding scores are calculated
    # Read Headline and Body files which have been produced after pre-processing
    training_sample= pd.read_csv('./post_processed/processed_train_sample.csv',sep = ',', encoding  = 'utf-8', index_col= 'id')

    # Main Body Collection
    # loads the body collection (body collection does not contain duplicate article records)
    body_collection = pd.read_csv('./post_processed/body_collection.csv', sep = ',', encoding  = 'utf-8', index_col= 'id')

    # RETRIEVAL MODELS
    # Vector Space Model - Calculating cosine similarity
    # VSM model - use the body Collection

    VSM = VSM()
    # load the collection's bodies
    VSM.load_collection(body_collection.iloc[:,:], 'Body')

    # Probabilistic Retrieval Models (Language Models, BM25)
    probabilistic_model = Probabilistic_models()
    probabilistic_model.load_collection(body_collection.iloc[:,:],'Body')

    # Step 4 - Calculates retrieval models scores
    # ------------------------------------------------------------------------------------------------------------------
    # Initialise required list
    dcosine_tf_idf,dcosine_tf,bms25,kld_jeliner,kld_dirichlet, kld_bayesian= [],[],[],[],[],[]

    print('Features/Attributes values are estimated...')
    for index in range(training_sample.shape[0]): #
        if index % 100 == 0:
            print(f'Iteration: {index}')
        # Similarity results
        # calculates cosine similarity for each pair of Headline - Body - tf*idf
        dcosine_tf_idf.append(VSM.cosine_similarity(training_sample.Headline[index],training_sample.Body[index], measure = 'tf_idf'))
        # calculates cosine similarity for each pair of Headline - Body - tf
        dcosine_tf.append(VSM.cosine_similarity(training_sample.Headline[index],training_sample.Body[index], measure='tf'))

        # Jeliner-mercer smoothing
        kld_jeliner.append(probabilistic_model.KL_divergence(training_sample.Headline[index], training_sample.Body[index],
                                                             method='jeliner-mercer', parameter=1, log=True))
        # Dirichlet
        kld_dirichlet.append(probabilistic_model.KL_divergence(training_sample.Headline[index], training_sample.Body[index],
                                                             method='dirichlet', parameter=0, log=True))
        # Bayesian
        kld_bayesian.append(probabilistic_model.KL_divergence(training_sample.Headline[index], training_sample.Body[index],
                                                             method='bayesian', parameter=0.5, log=True))
        # BM25 model
        bms25.append(probabilistic_model.BM25(training_sample.Headline[index], training_sample.Body[index], k1 =1.2 ,k2 = 0))

    # set a list of the unscaled list
    unscale_l = [kld_jeliner,kld_dirichlet,kld_bayesian,bms25]
    s_kld_jeliner,s_kld_dirichlet,s_kld_bayesian,s_bms25 = [scaling(unscale) for unscale in unscale_l]

    # exports the scores of the retrieval models
    pd.DataFrame({'tfxidf' : dcosine_tf_idf,'tf': dcosine_tf ,'BMS25' : s_bms25,'Jeliner' : s_kld_jeliner, 'Dirichlet': s_kld_dirichlet, 'Bayesian': s_kld_bayesian}).to_csv('./post_processed/scores/train_scores.csv', encoding = 'utf-8', sep = ',',index=False)

    # Step 5
    # ----------------------------------------------------------------------------------------------------------------------
    # Proceeds comparing the scores of each retrieval model
    # load scores results

    scores = pd.read_csv('./post_processed/scores/train_scores.csv', encoding = 'utf-8', )
    scores = pd.concat([scores, training_sample_raw['Stance']], axis=1)

    # Shows Distances/Features Distributions for BMS25 and Dirchlet Scores
    scores_comparison(scores,['BMS25','Dirichlet','Jeliner','Bayesian','tfxidf','tf'], n_plot= (2,3))

    # EXTRA STEP
    # ---------------------------------------------------------------------------------------------------------------------
    # Sensitivity Analysis based on boxcox parametric values. Scores were evaluated if they can approach a gaussian distribution

    # loop over the score values
    for score in [scores['BMS25'].values, scores['tfxidf'].values]:
        score_c = score
        # loop over an increament that may the scores needed (remove the influence of zero values)
        for increment in [e/10 for e in range(1,100)]:
            print(f'Increment: {increment}')
            # increase of the scores' values
            c = score_c + increment
            # loop over a lambda parameter which will determine the type of Box Cox transformation
            for lambda_par in [e/10 for e in range(-50,100)]:
                # transform the values based on a given lambda parameter
                t = boxcox(c,lmbda= lambda_par)
                # normalise the date so the mean should be zero with 1 standard deviation
                t_norm = (t - np.mean(t))/np.std(t)

                # Normality test (https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.mstats.normaltest.html)
                test_stat_normality = normaltest(t_norm)
                # Kolmogorov-Smirnov Test (https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.kstest.html)
                test_stat_ks = kstest(t_norm,'norm')
                # if the p-value of the statistic test is higher of 0.05, then the info is given
                # if a p-value is statistical significant, then show the parameter
                if test_stat_normality[1] > 0.05 or test_stat_ks[1] > 0.05:
                    print(f'Increament: {increament}')
                    print(f'Value: {lambda_par}\tStat: {test_stat}')

    # none values has indicated any significant sign of normality.
    # the process is continued without data that meet the assumption of normally distributed
