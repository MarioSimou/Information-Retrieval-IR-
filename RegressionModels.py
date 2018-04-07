import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib import style
from math import exp
from sklearn import linear_model

style.use('ggplot')

class regression():
    def __init__(self):
        pass
    def _get_shape(self, X):
        # Find the dimensions of the input variables
        try:
            shape = (X.shape[0], X.shape[1])
        except IndexError:
            shape = (1, X.shape[0])
        return shape

    def _generate_sse_plot(self,input, ylabel, title):
        # Plot the cost plot of the model in order to give insight how the error was varying
        # during the training of the model.
        if input != None:
            plt.plot(range(len(input)), input, color='red')
            plt.xlabel('$Epoch_i$')
            plt.ylabel('$' + ylabel + '$')
            plt.title(title)
            plt.show()

    def _predict(self, X_test,coefs,intercept):
        # set x as numpy array
        X_test = np.asarray(X_test)
        # Get the dimensions of the input variables

        shape = regression._get_shape(regression, X_test)
        # Checks if the input variables have the same dimensions as the trained model
        if shape[0] == len(coefs):
            # Simple Linear Regression
            if shape[0] == 1:
                # estimate yhat for n observations
                yhat = [intercept[0] + coefs[0] * x for x in X_test]
            # Multiple Linear Regression
            elif shape[0] != 1:
                # gets the intercept and coefficients
                intercept, coefs = intercept, coefs
                # zip the values of observation i
                observations = [e for e in zip(*X_test)]
                # estimate
                yhat = [intercept[0] + sum(list(map(lambda v, coef: v * coef, values, coefs))) for values in
                        observations]

            return np.asarray(yhat)
        else:
            raise Exception('The trained model does not have the same variables as the input X')

    def _plot_it(self,X,Y,yhat,shape,plot_it):
        if shape[0] == 1:
            if plot_it:
                plt.plot(X, yhat, color='red')
                plt.scatter(X, Y, color='blue')
                plt.show()
        else:
            print('Multiple regression model cannot display a 2D plot.')

    def _calculate_p_value(self,X,Y,yhat,coefs,n,shape):
        # Degrees of freedom
        df = n - 2
        # sample variance
        s = (((yhat - Y) ** 2).sum() / df) ** 0.5
        # sum of variable x squared differences
        if shape[0] == 1:
            Sxi = np.sqrt(shape[1]) * X.std()
        else:
            Sxi = [ np.sqrt(shape[1]) * x.std() for x in X]
        # standard error for each coefficient
        t_value = np.divide(coefs, np.divide(s, Sxi))
        p_value = 1 - stats.t.cdf(t_value, df=df)

        return p_value

    class linear_regression():
        @property
        def coefs(self):
            return self.__coefs
        @property
        def intercept(self):
            return self.__intercept
        @property
        def costs(self):
            return self.__sse_list
        @property
        def generate_sse_plot(self):
            regression._generate_sse_plot(regression,self.costs,'SSE(Iteration_i)','Epoch vs SSE')
        @property
        def p_values(self):
            return self.__p_values
        
        def predict(self,X_test):
            return regression._predict(regression,X_test,self.coefs,self.intercept)

        def fit(self,Y,X, method = 'stochastic', epochs = 100,epsilon = 0.0001, learning_rate = 0.01, plot_it = True):
            # Creates a numpy array for Y and X variable
            Y = np.asarray(Y)
            X = np.asanyarray(X)
            # assign the dimensions of X variable
            shape = regression._get_shape(regression,X)
            # Sample Length
            n = len(Y)
            # initialise converge variable
            converge = False

            # initialise the intercept and coefficients
            sse_list = []
            epoch = 0
            intercept = np.random.uniform(size  = 1) #np.asarray([0]) #np.random.uniform(size  = 1)
            coefs = np.random.uniform(size = shape[0]) #np.asarray([0]*shape[0])#np.random.uniform(size = shape[0])

            # initialise cost value and a cost list
            sse_initial= ((self.get_RMSE(Y,regression._predict(regression,X,coefs,intercept)))**2)/n


            if method == 'stochastic':
                while not converge:
                    for observation in range(shape[1]):
                        try:
                            if shape[0] == 1:
                                yhat = intercept[0] + coefs[0]* X[observation]
                            elif shape[0] != 1:
                                yhat = intercept[0] + sum([item[observation]*coefs[i] for i,item in enumerate(X)])

                            # current error - substraction between current real value and prediction
                            error = yhat - Y[observation]

                            # update intercept and coefficients
                            intercept = intercept - (error * learning_rate)
                            # Simple Linear Regression
                            if shape[0] ==1:
                                coefs = [coefs[0] - (error * learning_rate * X[observation])]
                            # Multiple Regression
                            elif shape[0] != 1:
                                coefs = [coefs[i] -(error * learning_rate * item[observation]) for i,item in enumerate(X)]

                        except Exception:
                           print(Exception.__doc__)

                    # sum of square errors of current epoch
                    sse = ((self.get_RMSE(Y,regression._predict(regression,X,coefs,intercept)))**2)/n
                    # add the sse error in the list
                    sse_list.append(sse)
                    # message
                    #print(f'Learning Rate: {learning_rate}\tEpoch: {epoch}\tSSE: {sse}')

                    # Conditions which will terminate the training of the model
                    if ((epoch >= epochs) or (abs(sse_initial- sse) < epsilon)):
                        converge = True

                    # update counter and sse_initial
                    epoch += 1 # epoch counter
                    sse_initial = sse # sum of square errors

                # calculates yhat for all Xs
                yhat = regression._predict(regression, X, coefs, intercept)
                # plots the regression resuls
                regression._plot_it(regression, X, Y, yhat, shape, plot_it)

            elif method == 'batch':
                while not converge:
                    if shape[0] == 1:
                        yhat = list(map(lambda v: intercept[0]+coefs[0]*v,X))
                    # Multiple Regression
                    elif shape[0] != 1:
                        # zip the values of observation i
                        observations = [e for e in zip(*X)]
                        # estimate
                        yhat = [intercept[0] + sum(list(map(lambda v, coef: v * coef, values, coefs))) for values in observations]


                    # update intercept
                    intercept = intercept - (learning_rate * (1/n*sum(list(map(lambda yhat,y: (yhat - y),yhat,Y)))))
                    # update coefficients
                    if shape[0] ==1:
                        # Linear Regression
                        coefs = [coefs[0] - (learning_rate * (1/n*sum(list(map(lambda yhat,y,x:(yhat-y)*x,yhat,Y,X)))))]
                    elif shape[0] != 1:
                        # Multiple Regression
                        # update the coefficient
                        coefs = [[coefs[0] - (learning_rate * (1/n*sum(list(map(lambda yhat,y,x:(yhat-y)*x,yhat,Y,x)))))] for x in X]
                        # unpack them from the list
                        coefs = [v for value in coefs for v in value]

                    # sum of square errors
                    sse = (self.get_RMSE(Y,regression._predict(regression, X, coefs, intercept)))**2/n
                    # adds the error to the cost values
                    sse_list.append(sse)

                    # message
                    #print(f'Learning Rate: {learning_rate}\tEpoch: {epoch}\tSSE: {sse}')

                    # Conditions which will terminate the training of the model
                    if ((epoch >= epochs) or (abs(sse - sse_initial) < epsilon)):
                        converge = True

                    # update epoch
                    epoch+=1
                    # update sse
                    sse_initial= sse

                # plots the regression resuls
                regression._plot_it(regression, X, Y, yhat, shape, plot_it)


            else:
                raise Exception('[Unrecognized method of optimization]')

            # calculates coefficients p-values
            p_values = regression._calculate_p_value(regression, X, Y, yhat, coefs, n, shape)

            # intercept, coefficients, cost_values and dimensions of the model are assigned as private variables
            self.__intercept = intercept
            self.__coefs = coefs
            self.__sse_list = sse_list
            self.__shape = shape
            self.__p_values = p_values


        def get_RMSE(self,Y,yhat):
            if len(Y) == len(yhat):
                # Calculates RMSE - > Sqrt(Sum((yhat - y)^2)/ny)
                return (sum(list(map(lambda yhat,y: (yhat-y)**2,yhat,Y)))/len(Y))**0.5
            else:
                raise Exception('[Different size of estimates]')

        def score(self,Y_test,X_test):
            shape = regression._get_shape(regression,X_test)
            # Number of Observations
            n = shape[0]
            # Number of Variables
            m = shape[1]
            # Y mean
            ym = np.asarray(Y_test).mean()
            # Yhat
            yhat = self.predict(X_test)
            # Sum of Squared Errors
            SSE = sum(list(map(lambda y_i,yhat_i: (y_i-yhat_i)**2,Y_test,yhat)))
            # Total Sum of Squares
            TSS = sum(list(map(lambda y_i: (y_i-ym)**2,Y_test)))
            # R squared
            Rsq = 1 - (SSE / TSS)
            # Adjusted R squared
            adj_Rsq = 1- (1-Rsq)*((m-1)/(m-n-1))

            # returns r square and adjusted r square
            return Rsq,adj_Rsq


    class logistic_model():

        @property
        def coefs(self):
            return self.__coefs
        @property
        def intercept(self):
            return self.__intercept
        @property
        def costs(self):
            return self.__cost_values
        @property
        def generate_accuracy_plot(self):
            regression._generate_sse_plot(regression,self.__accuracy_list, 'Accuracy \/\%', 'Epoch vs Accuracy')
        @property
        def p_values(self):
            return self.__p_values

        def predict(self,X_test):
            # predicted probabilities
            yhat = self.__sigmoid_function(regression._predict(regression,X_test,self.coefs,self.intercept))
            return yhat

        def fit(self, Y, X, epochs=1000, accuracy_threshold = 0.7, learning_rate=0.3, plot_it = True):
            # set the X and Y variables as numpy arrays
            Y = np.asarray(Y)
            X = np.asarray(X)

            # identifies the shape of the X variable
            shape = regression._get_shape(regression, X)
            # number of observations
            n = len(Y)
            # initialise converge variable
            converge = False
            # initialise cost value and a cost list
            # an initial value of 9999 is assigned for the cost value

            # initialise the counter and intercept and coefficients randomly
            epoch = 0
            intercept = np.random.uniform(size = 1)
            coefs = np.random.uniform(size = shape[0])
            accuracy_list= []

            while not converge:
                for observation in range(shape[1]):
                    try:
                        # estimate
                        if shape[0] == 1:
                            yhat = intercept + coefs[0] * X[observation]
                        elif shape[0] != 1:
                            yhat = intercept + sum([item[observation] * coefs[i] for i, item in enumerate(X)])

                        # probability of yhat
                        y_transformed = self.__sigmoid_function(yhat)

                        # update the coefficient
                        intercept = self.__update_coefficient(intercept, learning_rate, Y[observation], y_transformed,1)

                        # update intercept and coefficients
                        if shape[0] == 1:
                            coefs = [self.__update_coefficient(coefs[0],learning_rate,Y[observation],y_transformed,X[observation])]
                        elif shape[0] != 1:
                            coefs = [self.__update_coefficient(coefs[i],learning_rate,Y[observation],y_transformed,item[observation]) for i,item in enumerate(X)]

                    except:
                        raise Exception('Errorr')

                # accuracy of current epoch
                accuracy = self.accuracy(Y,self.__sigmoid_function(regression._predict(regression,X,coefs,intercept)))/100
                accuracy_list.append(accuracy)

                # message
                #print(f'Learning Rate: {learning_rate}\tEpoch: {epoch}\tAccuracy: {accuracy}')

                # Conditions which will terminate the training of the model
                if ((epoch >= epochs) or (accuracy >= accuracy_threshold)):
                    converge = True
                # epoch is updated
                epoch+= 1


            # plots the regression resuls
            regression._plot_it(regression, X, Y, regression._predict(regression, X, coefs, intercept), shape, plot_it)

            # intercept, coefficients, cost_values and dimensions of the model are assigned as private variables
            self.__intercept = intercept.tolist()
            self.__coefs = coefs
            self.__accuracy_list = accuracy_list
            self.__shape = shape

        def classifier(self,yhat,limit):
                if limit >0 and limit < 1:
                    # initialise zero and one variables
                    zero,one = 0,1
                    # transform probabilities values in zeros and ones
                    return  [ zero if prob < limit else one for prob in yhat]
                else:
                    raise Exception('The limit must be within the interval of 0 and 1.')

        def __sigmoid_function(self,yhat):
            if len(yhat) == 1:
                return 1/(1+exp(-yhat))
            else:
                return [1 / (1 + exp(-y)) for y in yhat]
        def __update_coefficient(self,b,learning_rate,y,prediction,x):
            return (b+learning_rate * (y-prediction) * prediction * (1-prediction) * x)
        def accuracy(self,Y,yhat):
            Y = np.asarray(Y)
            yhat = np.asarray(yhat)
            # number of values
            n = len(Y)
            # checks if the yhat is classified
            condition = n == (len(yhat == 1) + len(yhat == 0))
            if not condition:
                yhat = self.classifier(yhat, 0.5)
                correct_predictions = sum([ Y[i] == yhat[i] for i in range(len(yhat))])

                return (correct_predictions/float(n))*100

if __name__ == '__main__':

    x = np.array([1, 2, 4, 3, 5])  # x variable
    y = np.array([1, 3, 3, 2, 5])  # y variable


    # Creates a linear_regression object
    lm_stochastic = regression.linear_regression()
    # fit a model using stochastic gradient descent algorithm
    lm_stochastic.fit(y, x, method='stochastic', epochs=10000, epsilon= 0.000000001, plot_it=True, learning_rate= 0.001)
    # plot the sse plot over each epoch
    lm_stochastic.generate_sse_plot
    # prediction
    yhat_stochastic = lm_stochastic.predict(x)
    print(yhat_stochastic)
    print(f'\nStochastic Gradient Descent Info\nR square: {lm_stochastic.score(y,x)[0]}\nAdj R square: {lm_stochastic.score(y,x)[1]}\nCoeffs: {lm_stochastic.coefs}\tP-values: {lm_stochastic.p_values}\nIntercept: {lm_stochastic.intercept}\nRMSE: {lm_stochastic.get_RMSE(y,yhat_stochastic)}\n')

    lm_batch = regression.linear_regression()
    lm_batch.fit(y,x, method = 'batch', epochs= 10000,epsilon= 0.00000001, plot_it=True, learning_rate= 0.01)
    # plot the sse plot over each epoch
    lm_batch.generate_sse_plot
    yhat_batch = lm_batch.predict(x)
    print(yhat_batch)
    print(f'\nBatch Gradient Descent Info\nR square: {lm_batch.score(y,x)[0]}\nAdj R square: {lm_batch.score(y,x)[1]}\nCoeffs: {lm_batch.coefs}\tP-values: {lm_batch.p_values}\nIntercept: {lm_batch.intercept}\nRMSE: {lm_batch.get_RMSE(y,yhat_batch)}\n')

    # Check with sklearn models
    lm = linear_model.LinearRegression()
    lm.fit(x.reshape(-1,1),y.reshape(-1,1))
    print(f'\nSklearn Linear ModelInfo\nCoeffs: {lm.coef_}\nIntercept: {lm.intercept_}\n')

    # Logistic regression
    x = [2.7810836,1.465489372,3.396561688,1.38807019,3.06407232,7.627531214,5.332441248,6.922596716,8.675418651,7.673756466]
    y = [0,0,0,0,0,1,1,1,1,1]

    logistic = regression.logistic_model()
    logistic.fit(y,x, epochs=100, accuracy_threshold=1, learning_rate=0.3)
    print(f'Coefficients : {logistic.coefs}\tIntercept : {logistic.intercept}')
    logistic.generate_accuracy_plot
    yhat = logistic.predict(x)
    print(f'Classifiers : {logistic.classifier(yhat,0.5)}\nAccuracy: {logistic.accuracy(y,yhat)} %')
    

