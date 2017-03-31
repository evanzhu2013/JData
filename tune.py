
# Generic function for making a bagging classification model and accessing performance:

def fitting_model(model, data, predictors, outcome,cv=5):
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score
# Fit the model
    model.fit(data[predictors],data[outcome])

    # Make predictions on training set
    predictions = model.predict(data[predictors])

    # Print Train accuracy
    accuracy = accuracy_score(predictions,data[outcome])
    print("Training Accuracy:{0:.3%}".format(accuracy))

    scores = cross_val_score(model,data[predictors], data[outcome], cv=cv)
    print("Cross-Validation Score:{0:.3%}".format(scores.mean()))

    # Fit the model again so that it can be refered outside the function
    model.fit(data[predictors],data[outcome])

def Grid_search(model,param,data,pred_var,outcome_var,cv_results=False,beta=1):
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer
    from sklearn.metrics import fbeta_score, roc_auc_score
    scorer = make_scorer(fbeta_score,beta=beta)
    scorer_roc = make_scorer(roc_auc_score)
    # run randomized search
    grid_search = GridSearchCV(model, param_grid=param,scoring=scorer)
    grid_search.fit(data[pred_var],data[outcome_var])
    if cv_results==True:
        print("Model Rank:\n")
        print(report(grid_search.cv_results_))
    return grid_search.best_estimator_

def Random_search(model,param,data,pred_var,outcome_var,cv_results=False,n_iter_search=30):
    from sklearn.model_selection import RandomizedSearchCV
    # run randomized search
    random_search = RandomizedSearchCV(model, param_distributions=param,n_iter=n_iter_search)
    random_search.fit(data[pred_var],data[outcome_var])
    if cv_results==True:
        print("Model Rank:\n")
        print(report(random_search.cv_results_))
    return random_search.best_estimator_

def plot_test_auc(model,test,pred_var,outcome_var,label=None):
    from sklearn.metrics import roc_curve
    from matplotlib.pylab import plt
    y_score = model.predict_proba(test[pred_var])[:,1]
    fpr, tpr, thresholds = roc_curve(test[outcome_var],y_score)
    plt.plot(fpr,tpr,'b:',label=label)
    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1.03,0,1.03])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show

# Utility function to report best scores
def report(results, n_top=3):
    import numpy as np
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))

def test_accuracy(final_model,test,pred_var,outcome_var):
    from sklearn.metrics import precision_score,recall_score,roc_auc_score,confusion_matrix
    final_predictions = final_model.predict(test[pred_var])
    print("Accuracy:\n")
    print("Confusin matrix:")
    print(confusion_matrix(test[outcome_var],final_predictions))
    print("\n")
    print("Precision_score:{:.3f}".format(precision_score(test[outcome_var],final_predictions)))
    print("Recall_score:{:.3f}".format(recall_score(test[outcome_var],final_predictions)))
    print("ROC_Score:{:.3f}".format(roc_auc_score(test[outcome_var],final_predictions)))

def feature_plot(importances, X_train, y_train):

    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Creat the plot
    fig = plt.figure(figsize = (15,8))
    plt.title("Normalized Weights for First Five Most Predictive Features", fontsize = 16)
    plt.bar(np.arange(5), values, width = 0.6, align="center", color = '#00A000', \
          label = "Feature Weight")
    plt.bar(np.arange(5) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', label = "Cumulative Feature Weight")
    plt.xticks(np.arange(5), columns)
    plt.xlim((-0.5, 4.5))
    plt.ylabel("Weight", fontsize = 12)
    plt.xlabel("Feature", fontsize = 12)
    plt.legend(loc = 'upper center')
    plt.tight_layout()
    plt.show()
