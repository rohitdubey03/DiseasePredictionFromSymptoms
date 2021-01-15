# Disease-Prediction-from-Symptoms

This project is about prediction of disease based on symptoms using machine learning. Machine Learning algorithms such as Naive Bayes, Decision Tree and Random Forest are employed on the provided dataset and predict the disease. Its implementation is done through the python programming language. The research demonstrates the best algorithm based on their accuracy. The accuracy of an algorithm is determined by the performance on the given dataset.

# Dataset

The dataset for this problem is downloaded from here: 
```
https://impact.dbmi.columbia.edu/~friedma/Projects/DiseaseSymptomKB/index.html
```
## Training the models

After all the scraping and pre-processing, now it is time to rev up your enginesand do some magic (Not literally doing magic, just some extensive math) to train the machine learning models.

1. A binary vector is computed that consists of 1 for the symptoms present in the user’s selection list and 0 otherwise. A machine learning model is trained on the dataset, which is used here for prediction. The model accepts the symptom vector and outputs a list of top K diseases, sorted in the decreasing order of individual probabilities. As a common practice K is taken as 10. 

2. Multinomial Naïve Bayes, Random Forest, K-Nearest Neighbor, Logistic Regression, Support Vector Machine, Decision Tree were trained and tested with a train-test split of 90:10.

3. Multi layer Perceptron Neural Network was also trained and tested with the same split ratio.

4. You can find the implementation of all these models in main.py file.

5. Out of all these, Logistic Regression performed the best when tested against 5 folds of cross validation.

## Model training

1. Detected diseases using ML models
2. The probability of a disease is calculated as below.
3. ModelAccuracy->accuracy(model used)
4. DisASymp=Symptoms(DiseaseA)
5. match->intersect(DisASymp,userSymp)
6. matchScore->match/count(userSymp)
7. prob(DiseaseA)=matchScore * modelAccuracy
