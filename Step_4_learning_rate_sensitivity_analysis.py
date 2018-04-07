import pandas as pd
from matplotlib import pyplot as plt
from ir_package.RegressionModels import regression
import numpy as np
from sklearn import linear_model

# Functions
# ---------------------------------------------------------------------------------------------------------------------
def scatter_plot_custom(x1,y1,x2,y2):
    # Relation Plot
    plt.figure(1)
    plt.subplot(211)
    plt.scatter(x1,y1, color = 'red', label = 'BMS25')
    plt.xlabel('$Score_i$')
    plt.ylabel('$Stance_i$')
    plt.title('Score vs Stance')
    plt.legend(loc = 2)
    plt.subplot(212)
    plt.scatter(x2,y2, color = 'red', label = 'tfxidf')
    plt.xlabel('$Score_i$')
    plt.ylabel('$Category_i$')
    plt.legend(loc = 2)
    plt.show()

def df_adjustment(df):
    df['Categorical'] = df['Stance'].replace(['unrelated', 'disagree', 'discuss', 'agree'],[0.125, 0.375, 0.625, 0.875])
    df['c1'] = [1 if v == 'unrelated' else 0 for v in df.Stance.values]
    df['c2'] = [1 if v == 'disagree' else 0 for v in df.Stance.values]
    df['c3'] = [1 if v == 'discuss' else 0 for v in df.Stance.values]
    df['c4'] = [1 if v == 'agree' else 0 for v in df.Stance.values]

    return df
# ---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # STEP 1
    # ------------------------------------------------------------------------------------------------------------------
    # Train and Test Sample Load and Shuffle them
    train_sample = pd.concat([pd.read_csv('./FNC_sample/split_training_sample.csv', encoding='utf-8'), pd.read_csv('./post_processed/scores/train_scores.csv', encoding = 'utf-8', )], axis = 1).sample(frac = 1)
    test_sample = pd.concat([pd.read_csv('./FNC_sample/split_testing_sample.csv', encoding='utf-8'),pd.read_csv('./post_processed/scores/test_scores.csv', encoding = 'utf-8', )], axis = 1).sample(frac = 1)

    # BMS25 and tfxidf Modelling - Train/Test Dataframes
    # Train
    BMS25_train = pd.DataFrame({'Score' : train_sample['BMS25'], 'Stance' : train_sample['Stance']})
    tfxidf_train = pd.DataFrame({'Score' : train_sample['tfxidf'], 'Stance' : train_sample['Stance']})
    # Test
    BMS25_test = pd.DataFrame({'Score' : test_sample['BMS25'], 'Stance' : test_sample['Stance']})
    tfxidf_test = pd.DataFrame({'Score' : test_sample['tfxidf'], 'Stance' : test_sample['Stance']})

    # How the decision boundary is decided?
    # Update Stance Value
    # Unrelated = 1
    # Disagree = 2
    # Discuss = 3
    # Agree = 4

    # Transform Stance values in a numeric categorical variable
    # Train Samples
    BMS25_train = df_adjustment(BMS25_train)
    tfxidf_train = df_adjustment(tfxidf_train)
    # Test Samples
    BMS25_test = df_adjustment(BMS25_test)
    tfxidf_test = df_adjustment(tfxidf_test)

    # Test Plot
    scatter_plot_custom(BMS25_train.Score.values,BMS25_train.Stance.values,tfxidf_train.Score.values,tfxidf_train.Stance.values)
    # Train Plot
    scatter_plot_custom(BMS25_test.Score.values,BMS25_test.Stance.values,tfxidf_test.Score.values,tfxidf_test.Stance.values)

    # initial learning rates
    rate = [x/10000 for x in range(1,11)] + [0.005,0.01,0.05,0.1,0.2,0.3,0.4]
    error_list_linear,error_list_logistic, = [],[]

    # Linear Regression - Training
    for alpha in rate:
        print(f'Current Rate: {alpha}')
        for i,item in enumerate([(BMS25_train,BMS25_test,'BMS25'),(tfxidf_train,tfxidf_test,'tfxidf')]):
            #print(f'Processing {item[2]}...')
            # Retrieval Model
            X_train,Y_train = item[0].Score.values,item[0].Categorical.values
            X_test,Y_test,Y_test_labels= item[1].Score.values,item[1].Categorical.values,item[1].Stance.values
            # Check

            lm =  linear_model.LinearRegression()
            lm.fit(np.asarray(X_train).reshape(-1,1),np.asarray(Y_train).reshape(-1,1))
            print(lm.get_params())
            print(lm.coef_)
            print(lm.intercept_)

            # LINEAR REGRESSION
            linear_m = regression.linear_regression()
            # fit the model using the training sample
            linear_m.fit(Y_train,X_train, method= 'stochastic',  epochs = 1000, epsilon = 0.000000001,learning_rate = alpha, plot_it= False)
            #linear_m.generate_sse_plot
            # prediction
            yhat_linear = linear_m.predict(X_test)
            # classifies the predictions under the hypothesis that:
            # h(theta) < 0.25 = 'unrelated'
            # h(theta) >= 0.25 and h(theta) < 0.5 = 'disagree'
            # h(theta) >= 0.5 and h(theta) < 0.75 = 'discuss'
            # h(theta) >= 0.75  = 'agree'
            linear_classifier = [0.125 if x < 0.25 else 0.375 if x >=0.25 and x < 0.5 else 0.625 if x < 0.75 and x >= 0.5 else 0.875 for x in yhat_linear]
            # finds the error rate of the linear classifier
            correct_linear = sum([linear_classifier[i] == Y_test[i] for i in range(len(yhat_linear))])
            # accuracy -Linear Regression
            error_rate_linear = 1-(correct_linear/len(Y_test))
            #print(f'Intercept: {linear_m.intercept}\tCoefficients: {linear_m.coefs}\tP-value: {linear_m.p_values}')

            # Logistic Regression
            # classifier 1 P( y = 'unrelated' = 1 | x;theta), P( y = 'disagree' or 'discuss' or 'agree' = 0 | x;theta)
            c1= regression.logistic_model()
            c1.fit(item[0].c1.values, item[0].Score.values, epochs= 1000, learning_rate= alpha, accuracy_threshold= 0.9, plot_it= False)
            # classifier 2 P( y = 'disagree' = 1 |x;theta), P( y = 'unrelated' or 'discuss' or 'agree' = 0 | x; theta)
            c2= regression.logistic_model()
            c2.fit(item[0].c2.values, item[0].Score.values, epochs= 1000, learning_rate= alpha, accuracy_threshold= 0.9,plot_it= False)
            # classifier 3 P( y = 'discuss' = 1 |x;theta), P( y = 'unrelated' or 'disagree' or 'agree' = 0 | x; theta)
            c3 = regression.logistic_model()
            c3.fit(item[0].c3.values, item[0].Score.values, epochs= 1000, learning_rate= alpha, accuracy_threshold= 0.9, plot_it= False)
            # classifier 4 P( y = 'agree' = 1 |x;theta), P( y = 'unrelated' or 'disagree' or 'discuss' = 0 | x; theta)
            c4 = regression.logistic_model()
            c4.fit(item[0].c4.values, item[0].Score.values, epochs= 1000, learning_rate= alpha, accuracy_threshold= 0.9, plot_it= False)

            #print(f'Performing prediction for Logistic regression {item[2]}')
            # probability for classifier 1
            p1 = c1.predict(item[1].Score.values)
            # probability for classifier 2
            p2 = c2.predict(item[1].Score.values)
            # probability for classifier 3
            p3 = c3.predict(item[1].Score.values)
            # probability for classifier 4
            p4 = c4.predict(item[1].Score.values)

            probabilities = [x for x in zip(p1,p2,p3,p4)]
            classification_list = []
            for l in probabilities:
                index = l.index(max(l))
                if index == 0:
                    stance = 'unrelated'
                elif index == 1:
                    stance = 'disagree'
                elif index == 2:
                    stance = 'discuss'
                elif index ==3:
                    stance = 'agree'
                classification_list.append(stance)

            # Accuracy
            # estimates the correct predictions
            correct_logistic = sum(list(map(lambda yhat,y : yhat == y, classification_list,Y_test_labels)))
            # finds the accuracy of the model
            error_rate_logistic = 1-(correct_logistic / len(Y_test_labels))
            # store to the list
            error_list_linear.append(error_rate_linear)
            error_list_logistic.append(error_rate_logistic)

    x = [str(x) for x in rate]
    # Linear and Logistic Classifier for BMS25 and tfxidf
    error_rate_bms25_linear = [error_list_linear[i] for i in range(len(error_list_linear)) if i%2 == 0 ]
    error_rate_bms25_logistic = [error_list_logistic[i] for i in range(len(error_list_logistic)) if i%2 == 0 ]
    error_rate_tfxidf_linear = [error_list_linear[i] for i in range(len(error_list_linear)) if i%2 != 0 ]
    error_rate_tfxidf_logistic = [error_list_logistic[i] for i in range(len(error_list_logistic)) if i%2 != 0 ]

    # File Exporting
    pd.DataFrame({'Rate' : rate, 'Error Linear BMS' :  error_rate_bms25_linear, 'Error Logistic BMS 25': error_rate_bms25_logistic,'Error Linear tfxidf' :  error_rate_tfxidf_linear, 'Error Logistic tfxidf': error_rate_tfxidf_logistic}).\
        to_csv('./sensitivity/learning_rate_parameters_sensitivity_analysis.csv', encoding= 'utf-8', sep =',')
    
    # STEP 2
    #------------------------------------------------------------------------------------------------------------------
    # Plotting The results
    # x axis
    rate = [x/10000 for x in range(1,11)] + [0.005,0.01,0.05,0.1,0.2,0.3,0.4]
    x = [str(x) for x in rate]
    
    # y axis
    results = pd.read_csv('./sensitivity/learning_rate_parameters_sensitivity_analysis.csv')
    err_bms25_linear, err_bms_logistic = results['Error Linear BMS'],results['Error Logistic BMS 25']
    err_tfxidf_linear, err_tfxidf_logistic = results['Error Linear tfxidf'],results['Error Logistic tfxidf']
    
    # Plot accuracy results
    figure = plt.gcf() # get current figure
    figure.set_size_inches(16, 12)
    plt.title('$\mathbf{Learning\/Rate\/vs\/Error\/Rate\/(ERR)}$')
    plt.plot(x, err_bms25_linear, color = 'red',label = 'BMS25 Linear')
    plt.plot(x, err_bms_logistic, color = 'red', linestyle = '--',label = 'BMS25 Logistic ')
    plt.plot(x, err_tfxidf_linear, color = 'blue',label = 'tfxidf Linear')
    plt.plot(x, err_tfxidf_logistic, color = 'blue',linestyle = '--', label = 'tfxidf Logistic ')
    plt.xlabel('$Learning\/Rate$')
    plt.ylabel('$Error\/Rate\/(ERR)$')
    plt.legend(loc = 1)
    plt.savefig('./img/Linear_classifier_plot.png', dpi = 200)
    plt.show()

