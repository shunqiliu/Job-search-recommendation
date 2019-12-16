# Recommend jobs by given profile
## This project more like the screening resumes from Linkedin
## The project mainly include word2vec, naive bayes
## Using the genism library as the baseline (contrast)

The raw data comes from Upenn courses, which cannot shown to public.

The genism model comes from GoogleNews-vectors-negative300.bin, which can be found at: https://code.google.com/archive/p/word2vec/. And a kaggle project: https://www.kaggle.com/rtatman/how-semantically-similar-are-two-job-titles

## jr_baseline
jr_baseline.py is the code about baseline method and adding labels on unsupervised data.

Each resume has 13 features(elements): locality, industry, summary, specilities, interests, major, desc, degree, member, affilition, univ name. 
Divide whole data to 20:3 (supervised data: unsupervised data). Compare the similarity of words in those features, assign jobtitle from supervised data to unsupervised data with highest confidence(similarity).
Compare the result with true labels.

##
