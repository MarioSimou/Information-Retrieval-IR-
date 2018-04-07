from ir_package.VSM import VSM
from ir_package.ProbabilisticModels import Probabilistic_models
from Step_1_FNC_preprocessing_scores import scaling,KL_divergence
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

'''
# Read Headline and Body files which have been produced after pre-processing
training_sample= pd.read_csv('./post_processed/processed_train_sample.csv',sep = ',', encoding  = 'utf-8', index_col= 'id')

# Main Body Collection
# loads the body collection (body collection does not contain duplicate article records)
body_collection = pd.read_csv('./post_processed/body_collection.csv', sep = ',', encoding  = 'utf-8', index_col= 'id')

# creates the VSM model and load the body collection (this collection does not include dupliate article bodies)
VSM = VSM()
# load the collection's bodies
VSM.load_collection(body_collection.iloc[0:10,:], 'Body')

# creates a probabilistic model and load the body article collection (Language Models, BM25)
probabilistic_model = Probabilistic_models()
probabilistic_model.load_collection(body_collection.iloc[0:10,:],'Body')

print('Features/Attributes values are estimated...')

# FIRST TRIAL
# Numerical values of smoothing parameters
range_jeliner = [x/10.0 for x in range(0,11)]
range_dirichlet_bayesian = range(0,220,20)

for i_range,item in enumerate(range_dirichlet_bayesian):
    # Initialise required list
    bms25, kld_jeliner, kld_dirichlet, kld_bayesian = [], [], [], [], [], []
    print(f'Iteration : {i_range} of {len(range_dirichlet_bayesian)}')
    for index in range(training_sample.shape[0]): #
        # Jeliner-mercer smoothing
        kld_jeliner.append(probabilistic_model.KL_divergence(training_sample.Headline[index], training_sample.Body[index],
                                                             method='jeliner-mercer', parameter=range_jeliner[i_range], log=True))
        # Dirichlet
        kld_dirichlet.append(probabilistic_model.KL_divergence(training_sample.Headline[index], training_sample.Body[index],
                                                             method='dirichlet', parameter=item, log=True))
        # Bayesian
        kld_bayesian.append(probabilistic_model.KL_divergence(training_sample.Headline[index], training_sample.Body[index],
                                                             method='bayesian', parameter=item, log=True))
        # BM25 model
        bms25.append(probabilistic_model.BM25(training_sample.Headline[index], training_sample.Body[index], k1 =1.2 ,k2 = item))

    # set a list of the unscaled list
    unscale_l = [kld_jeliner,kld_dirichlet,kld_bayesian,bms25]
    s_kld_jeliner,s_kld_dirichlet,s_kld_bayesian,s_bms25 = [scaling(unscale) for unscale in unscale_l]
    
    # set a list of the unscaled list
    unscale_l = [kld_dirichlet,bms25]
    s_kld_dirichlet,s_bms25 = [scaling(unscale) for unscale in unscale_l]

    # Results Exporting
    pd.DataFrame({'BMS25' : s_bms25,'Jeliner' : s_kld_jeliner, 'Dirchlet': s_kld_dirichlet, 'Bayesian': s_kld_bayesian}).to_csv('./sensitivity/scores_'+ str(i_range) +'.csv', encoding = 'utf-8', sep = ',',index=False)
'''
# Distances Distribution
def scores_comparison(df,names,width = 0.25, n_plot = (1,2)):
    x_ref = np.asarray([0.875, 0.625, 0.375, 0.125])
    y_ref_prob = [0.0736,0.1783,0.0168,0.74]

    i,j = 0,0
    f, axarr = plt.subplots(n_plot[0],n_plot[1])
    kl_divergence_value = []
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
        # kl divergence values
        v = KL_divergence(y_ref_prob, y_prob)
        # adds the score to the corresponded list
        kl_divergence_value.append(v)
        print(f'{name} - KL Divergence Value : {v}')
        try:
            axarr[i, j].bar(x_ref,y_ref_prob, width, fill = False, edgecolor = ['red','red','red','red'], label = 'reference')
            axarr[i, j].bar(x_ref,y_prob, width ,color = 'green', alpha = 0.5, label = name)
            axarr[i, j].set_title(name + '$\/vs\/Reference\/Histogram$')
            axarr[i, j].set_xticks(x_ref,['Agree','Discuss','Disagree','Unrelated'])
            axarr[i, j].set_ylabel('$P(Stance_i)$')
            axarr[i, j].set_xlabel('$Stance_i$')
            axarr[i, j].legend(loc = 1)
        except:
            axarr[j].bar(x_ref, y_ref_prob, width, fill=False, edgecolor=['red', 'red', 'red', 'red'], label='reference')
            axarr[j].bar(x_ref, y_prob, width, color='green', alpha=0.5, label=name)
            axarr[j].set_title(name + '$\/vs\/Reference\/Histogram$')
            axarr[j].set_xticks(x_ref, ['Agree', 'Discuss', 'Disagree', 'Unrelated'])
            axarr[j].set_ylabel('$P(Stance_i)$')
            axarr[j].set_xlabel('$Stance_i$')
            axarr[j].legend(loc=1)
        j+=1

    # Fine-tune figure; make subplots farther from each other.
    f.subplots_adjust(hspace=0.3)
    # save the sensitivity analysis scores
    plt.show()

    l1.append(kl_divergence_value[0])
    l2.append(kl_divergence_value[1])
    l3.append(kl_divergence_value[2])
    l4.append(kl_divergence_value[3])

if __name__ == '__main__':
    l1,l2,l3,l4 =[],[],[],[]
    for i in range(11):
        scores = pd.read_csv('./sensitivity/scores_'+ str(i)+'.csv', encoding='utf-8')
        scores_comparison(scores,['BMS25','Bayesian','Dirichlet','Jeliner'], n_plot=(2,2))
    
    range_jeliner = [x / 10.0 for x in range(0, 11)]
    range_dirichlet_bayesian = range(0,220,20)
    plt.figure(1)
    plt.subplot(211)
    plt.title('$\mathbf{Measuring\/Smoothing\/Parameters}$')
    plt.plot(range_dirichlet_bayesian,l1, color = 'red',label = '$BMS25$')
    plt.plot(range_dirichlet_bayesian,l2, color = '#f44e42',label = '$Bayesian$', linestyle = '--')
    plt.plot(range_dirichlet_bayesian,l3, color = '#471b18',label = '$Dirichlet$', linestyle = '-.')
    plt.xlabel('$Smoothing Parameters$')
    plt.ylabel('$D_KL(P|Q)$')
    plt.legend(loc = 2)
    plt.subplot(212)
    plt.plot(range_jeliner, l4, label='$Jeliner$',color = '#f44e42', linestyle = '--')
    plt.xlabel('$Smoothing\/Parameters$')
    plt.ylabel('$D_KL(P|Q)$')
    plt.legend(loc = 2)
    plt.savefig('./img/distances_retrieval_models.png')
    plt.show()
