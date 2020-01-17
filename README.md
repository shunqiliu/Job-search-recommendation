# Recommend jobs by given profile
Job hunting can be a tedious process for both job seekers and recruiters. On one hand, dealing with huge amount of job application, recruiters spend hours to look at applicants’ profiles and determine if they are qualified for the second round. On the other hand, many newly graduates do not know what job they want to seek for and the chance of success on application for a certain job title. 

The main idea of this project is to implement the function: screening resumes. Using machine learning method to identify if a given profile is suitable for a certain job position. Instead of giving a definitive identification, the recommend system shows the probability of "adaptability", as a reference for recruiters to make judgments, thus help both recruiters and job seekers evaluate if a certain job title is suitable for a person given his or her backgrounds. 

The result of the paper is that we achieve accuracies that are at least 80%, which is much larger than the baseline accuracy. Also, the fuzzy label Bayes gives both confidence and predictions successfully.

The project includes word2vec, naive bayes. Using the genism library as the baseline (contrast)

The raw data comes from Upenn courses, which cannot shown to public.

The genism model comes from GoogleNews-vectors-negative300.bin, which can be found at: https://code.google.com/archive/p/word2vec/. 
And the kaggle project: https://www.kaggle.com/rtatman/how-semantically-similar-are-two-job-titles

## Dataset Preparation
For our training dataset, we used a LinkedIn dataset which consists of 100000 people’s linkedin profile. For each person, we selected the following attributes: locality, industry, summary, specialties, interests, major, education, degree, affiliation. For the ease of our training, we dropped entries with missing values and the result is that for our final training set, we have 5701 entries with the above values. We selected people with their latest experience’ title as raw label. 

## Label Processing
As the labels extracted from data are characters of job titles, we want to transfer them to {0,1} labels. One way to do that is to compare the similarity of target job title and label job titles. The labeling rule is: 0, for similarity<50%; 1 for imilarity≥50%.

Calculation of similarity is based on build in function of gensim.model.similarity. Also, the models in gensim are trained by word2vec and the text from raw data

## Baseline
Baseline is used for comparing the accuracy of non-machine learning method and machine learning method. To give the reason why we should use machine learning to solve this problem.

Our baseline is based on the similarity of keywords in each attribute, like position, degree, etc. We firstly compute the score, which is weight sum of each attribute, of each testing data with all labeled data (training data). Then assign the highest score training data label (job title) to the testing data. Finally compute the accuracy. 

baseline.py is the code about baseline method and adding labels on unsupervised data.

## Machine Learning Classifier
As suggested above, we tried to beat the baseline model using selected machine learning classifiers. Prior to each model, we run a preliminary test for hyperparameter selection. 

For each classifier, we reckon that the amount of useful information is related to the number of keys we used to featurelize. Also, we want to explore how many instances we need to achieve a reasonable accuracy. Thus, we sweep number of keys included and the number of instances included for each classifier to get an accuracy overview ([50, 100, 250, 500, 1000, 2500, 5000])

classifier.py is the code about constructing multiple classifiers, training/testing data, and giving the result. 

## Fuzzy Label Naive-Bayes
As we generate {0,1} labels with uncertainty, it is a better way to maintain the probabilities of labels, which we call them confidence. We retrain the naive Bayes classifier with those fuzzy labels, by considering the P(y) and P(v|y) are changed by the confidence of labels, we build a new naive Bayes model.

In prediction part, we use the new model to do prediction on testing data. Furthermore, the accuracy here can be used as the confidence of the model, which means the prediction based on this model has some probability of correctness. This probability of correctness can give advice for recruiter about the prediction.

fuzzy label bayes.py is the code about constructing fuzzy label naive-bayes model and giving the result.

## Conclusion
After implementing all the machine learning classifiers above, we achieve accuracies that are at least 80% which is much larger than the baseline accuracy. Among these methods, Neural Network performs best with around 95% accuracy and robustness to noise. Besides, we introduce fuzzy label Naïve-Bayes as an extra criterion for judgement. Fuzzy label Naïve-Bayes provides average confidence for whole database and evaluating how good the label and features we constructed which is 57% in our case. In conclusion, the Resume screening system we built based on machine learning can help HR filtering job seekers at the very first moment without necessity to look through every resume. It can also help job hunters assessing themselves before they submit their resumes.

## job recommendation report.pdf 
job recommendation report.pdf is the report of the project, you can find any detail about this project in this file.
