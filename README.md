# Recommend jobs by given profile
## This project more like the screening resumes from Linkedin
## The project mainly include word2vec, naive bayes
## Using the genism library as the baseline (contrast)

The raw data comes from Upenn courses, which cannot show it to public.

The genism model comes from GoogleNews-vectors-negative300.bin, which can be found at: https://code.google.com/archive/p/word2vec/. And a kaggle project: https://www.kaggle.com/rtatman/how-semantically-similar-are-two-job-titles

jr.py is the main code about generating baseline and adding labels on unsupervised data. Here labels are fuzzy lables, which means the labels with probability.
