import pandas as pd
from ir_package.RegressionModels import regression
from Step_4_learning_rate_sensitivity_analysis import df_adjustment
import numpy as np
from sklearn import linear_model

if __name__ == '__main__':
    # Step 1 - FILE LOADING
    # ----------------------------------------------------------------------------------------------------------------------
    # Train and Test Sample Load and Shuffle them
    train_sample = pd.concat([pd.read_csv('./FNC_sample/split_training_sample.csv', encoding='utf-8'),
                              pd.read_csv('./post_processed/scores/train_scores.csv', encoding='utf-8', )], axis=1)
    test_sample = pd.concat([pd.read_csv('./FNC_sample/split_testing_sample.csv', encoding='utf-8'),
                             pd.read_csv('./post_processed/scores/test_scores.csv', encoding='utf-8', )], axis=1)

    # BMS25 and tfxidf Modelling - Train/Test Dataframes
    # Train
    BMS25_train = pd.DataFrame({'Score': train_sample['BMS25'], 'Stance': train_sample['Stance']})
    tfxidf_train = pd.DataFrame({'Score': train_sample['tfxidf'], 'Stance': train_sample['Stance']})
    # Test
    BMS25_test = pd.DataFrame({'Score': test_sample['BMS25'], 'Stance': test_sample['Stance']})
    tfxidf_test = pd.DataFrame({'Score': test_sample['tfxidf'], 'Stance': test_sample['Stance']})

    # Transform Stance values in a numeric categorical variable
    # Train Samples
    BMS25_train = df_adjustment(BMS25_train)
    tfxidf_train = df_adjustment(tfxidf_train)
    # Test Samples
    BMS25_test = df_adjustment(BMS25_test)
    tfxidf_test = df_adjustment(tfxidf_test)

    # STEP 2 - MODELS COMPARISON
    # ----------------------------------------------------------------------------------------------------------------------

    # create empty dataframes, which will contain the results of sklearn and custom linear models
    # the models are checked for BMS25 and tfxidf scores
    df_linear = pd.DataFrame({'BMS25' : [],'BMS25_sklearn' : [],'tfxidf' : [],'tfxidf_sklearn' : []} )
    df_logistic = pd.DataFrame({'BMS25' : [],'BMS25_sklearn' : [], 'tfxidf' : [],'tfxidf_sklearn' : []})

    # Learning Rate that is used:
    #           Linear  Logistic
    # BMS25     0.4       0.0002
    # tfxidf    0.4       0.01
    for j in range(0, 30):
        print(f'i: {j}')
        for i, item in enumerate([(BMS25_train.sample(frac=1), BMS25_test.sample(frac=1), 'BMS25',[0.4,0.0002]),(tfxidf_train.sample(frac=1), tfxidf_test.sample(frac=1), 'tfxidf',[0.4,0.01])]):
            # print(f'Processing {item[2]}...')
            # Split validation training and validation subsamples
            X_train, Y_train = item[0].Score.values, item[0].Categorical.values
            X_test, Y_test, Y_test_labels = item[1].Score.values, item[1].Categorical.values, item[1].Stance.values

            # Construct the Linear Regression model of sklearn package and checks it
            lm = linear_model.LinearRegression()
            lm.fit(np.asarray(X_train).reshape(-1, 1), np.asarray(Y_train).reshape(-1, 1))
            #print(f'SKLEARN model\tIntercept: {lm.intercept_}\tCoefficients: {lm.coef_}')

            # Construct the custom Linear Regression and checks it
            linear_m = regression.linear_regression()
            # fit the model using the training sample
            linear_m.fit(Y_train, X_train, method='stochastic', epochs=10000, epsilon=0.000000001, learning_rate=0.4, plot_it=False)
            #linear_m.generate_sse_plot

            # test the validation subsample
            yhat_linear_custom = linear_m.predict(X_test)
            yhat_linear_sklearn = np.asarray([item for list_item in lm.predict(np.asarray(X_test).reshape(-1,1)) for item in list_item])

            # classifies the predictions under the following thresholds:
            # h(theta) < 0.25 = 'unrelated'
            # h(theta) >= 0.25 and h(theta) < 0.5 = 'disagree'
            # h(theta) >= 0.5 and h(theta) < 0.75 = 'discuss'
            # h(theta) >= 0.75  = 'agree'

            linear_classifier = [[0.125 if x < 0.25 else 0.375 if x >= 0.25 and x < 0.5 else 0.625 if x < 0.75 and x >= 0.5 else 0.875 for x in list_item] for list_item in [yhat_linear_custom,yhat_linear_sklearn]]

            # calculates the correct predictions for both linear regression models
            correct_linear = [sum([linear_classifier[i][j] == Y_test[j] for j in item]) for i,item in enumerate([range(len(yhat_linear_custom)),range(len(yhat_linear_sklearn))])]
            # estimates the error rate for both Linear Regression models
            error_rate_linear = [1 - (item / len(Y_test)) for item in correct_linear]

            #print(f'Custom Linear Model\tIntercept: {linear_m.intercept}\tCoefficients: {linear_m.coefs}\tP-value: {linear_m.p_values}')
            #print(f'ERR Custom: {error_rate_linear[0]}\tERR sklearn: {error_rate_linear[1]}')
            # data are appended to the dataframe
            df_linear.at[j,item[2]] = error_rate_linear[0]
            df_linear.at[j, item[2] + '_sklearn'] = error_rate_linear[1]

            # LOGISTIC REGRESSION MODEL
            # classifier 1 P( y = 'unrelated' = 1 | x;theta), P( y = 'disagree' or 'discuss' or 'agree' = 0 | x;theta)
            c1 = regression.logistic_model()
            c1.fit(item[0].c1.values, item[0].Score.values, epochs=1000, learning_rate=item[3][1], accuracy_threshold=0.9, plot_it=False)
            # sklearn package model
            c1_sklearn = linear_model.LogisticRegression()
            c1_sklearn.fit(item[0].Score.values.reshape(-1,1),item[0].c1.values)

            # classifier 2 P( y = 'disagree' = 1 |x;theta), P( y = 'unrelated' or 'discuss' or 'agree' = 0 | x; theta)
            c2 = regression.logistic_model()
            c2.fit(item[0].c2.values, item[0].Score.values, epochs=1000, learning_rate=item[3][1], accuracy_threshold=0.9, plot_it=False)

            # sklearn package model
            c2_sklearn = linear_model.LogisticRegression()
            c2_sklearn.fit(item[0].Score.values.reshape(-1, 1), item[0].c2.values)

            # classifier 3 P( y = 'discuss' = 1 |x;theta), P( y = 'unrelated' or 'disagree' or 'agree' = 0 | x; theta)
            c3 = regression.logistic_model()
            c3.fit(item[0].c3.values, item[0].Score.values, epochs=1000, learning_rate=item[3][1], accuracy_threshold=0.9, plot_it=False)

            # sklearn package model
            c3_sklearn = linear_model.LogisticRegression()
            c3_sklearn.fit(item[0].Score.values.reshape(-1, 1), item[0].c3.values)

            # classifier 4 P( y = 'agree' = 1 |x;theta), P( y = 'unrelated' or 'disagree' or 'discuss' = 0 | x; theta)
            c4 = regression.logistic_model()
            c4.fit(item[0].c4.values, item[0].Score.values, epochs=1000, learning_rate=item[3][1], accuracy_threshold=0.9, plot_it=False)

            # sklearn package model
            c4_sklearn = linear_model.LogisticRegression()
            c4_sklearn.fit(item[0].Score.values.reshape(-1, 1), item[0].c4.values)

            # Test the validation subsample
            # probability for classifier 1
            p1 = c1.predict(item[1].Score.values)
            # probability for classifier 2
            p2 = c2.predict(item[1].Score.values)
            # probability for classifier 3
            p3 = c3.predict(item[1].Score.values)
            # probability for classifier 4
            p4 = c4.predict(item[1].Score.values)

            # sklearn
            # probability for classifier 1
            p1_sklearn = np.asarray(c1_sklearn.predict(np.asarray(item[1].Score.values).reshape(-1,1))) #c1_sklearn.predict(item[1].Score.values)
            # probability for classifier 2
            p2_sklearn = np.asarray(c2_sklearn.predict(np.asarray(item[1].Score.values).reshape(-1,1)))
            # probability for classifier 3
            p3_sklearn = np.asarray(c3_sklearn.predict(np.asarray(item[1].Score.values).reshape(-1,1)))
            # probability for classifier 4
            p4_sklearn = np.asarray(c4_sklearn.predict(np.asarray(item[1].Score.values).reshape(-1,1)))

            # zip the probabilitie
            probabilities,probabilities_sklearn = [x for x in zip(p1, p2, p3, p4)],[x for x in zip(p1_sklearn, p2_sklearn, p3_sklearn, p4_sklearn)]
            # empty classification leasts
            classification_list,classification_list_sklearn = [],[]

            # classifies the probabilities to the correspond class
            # for an input x, the classifier that has the highest probability is considered as the most possible class of the input
            for probability_item in [(probabilities,classification_list),(probabilities_sklearn,classification_list_sklearn)]:
                for l in probability_item[0]:
                    index = l.index(max(l))
                    if index == 0:
                        stance = 'unrelated'
                    elif index == 1:
                        stance = 'disagree'
                    elif index == 2:
                        stance = 'discuss'
                    elif index == 3:
                        stance = 'agree'
                    probability_item[1].append(stance)

            # Calculating the accuracy
            # estimates the correct predictions
            correct_logistic = [sum(list(map(lambda yhat, y: yhat == y, item, Y_test_labels))) for item in [classification_list,classification_list_sklearn]]
            # finds the accuracy of the model
            error_rate_logistic = [1 - (item / len(Y_test_labels)) for item in correct_logistic]

            # store the results at the corresponding dataframe
            df_logistic.at[j, item[2]] = error_rate_logistic[0]
            df_logistic.at[j, item[2]+ '_sklearn'] = error_rate_logistic[1]

            # 30 iterations need to be performed

    # Print Results of Linear Classifier
    print(f'Linear Regression Classifier\n{df_linear.mean(axis = 0)}')
    print(df_linear.std(axis = 0))
    # Prints Results of Logistic Classifier
    print(f'Logistic Regression Classifier\n{df_logistic.mean(axis = 0)}')
    print(df_logistic.std(axis = 0))
