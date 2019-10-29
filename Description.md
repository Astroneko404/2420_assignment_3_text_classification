#### HOMEWORK 3 (CS 2731)
##### Assigned: October 22, 2019
##### Due: November 5, 2019 (before midnight)
This assignment provides hands-on experience with 1) applying baseline machine learning methods 
for a text classification task using bag-of-words and vector semantics, and 2) posing a research 
question and setting up an experiment to address the question. 
To do so, you will develop classifiers for toxicity of user comments to news articles, using the 
annotated constructiveness and toxicity corpus from the SFU opinions and comments corpus 
(if you want more background, click here). In particular, given a comment from Column F, predict 
the level of toxicity (use the left-most/first number in Column I of the corresponding row).

#### Main Tasks
1. Set up your global experimental framework. Make appropriate cross-validation splits. 
[NB: you should think about how you want to randomize the data and be able to justify your choice 
in the write-up. Note that in the file, the instances are sorted by the original articles and then 
by comment order.]
2. Build a baseline logistic regression classifier and compare with a majority-vote classifier:
    * First, extract and preprocess the comment text so as to determine your vocabulary set for 
    this task.
    * Next, train a logistic regression classifier using bag of words. You may use standard 
    off-the-shelf packages for training the classifier.
    * Finally, record the performance of your logistic regression classifier using cross-validation.
    How does it compare against a majority-vote baseline? 
3. Make two improvements to your logistic regression baseline classifier and compare with the 
previous classifier:
    * One by moving from bag of words to a sparse vector semantic representation.
    * And another from bag of words to a dense vector semantic representation.
    * Describe and compare the changes to each classifier made from Step 2. 
4. Perform a rigorous comparison between your 3 classifiers using statistical tests.
5. Pose a simple question based on this classification task; conduct an experiment to answer the 
question; discuss the outcomes of the experiment and draw conclusions.
    * This portion is intended to be more open-ended and exploratory. You are encouraged to not 
    pose the exact same question as someone else.
    * You may frame your question to explore any of the issues that we have discussed thus far 
    where multiple plausible options exist (e.g., text normalization, sentiment analysis, 
    cross-validation, vector representations, discriminative/generative ML). 
6. Unlike prior homeworks, you are allowed to use external resources, including:
    * Standard off-the-shelf packages such as: NLTK, Stanford CoreNLP, SciKit.
    * Pre-trained word embeddings 
#### What to Submit
* Your code and data files
    * Please document enough of your program to help the TA grade your work. 
* A README file that addresses the following:
    * Describe the computing environment you used, especially if you used some off-the-shelf modules. (Do not use unusual packages. If you're not sure, please ask.)
    * List any additional resources, references, or web pages you've consulted.
    * List any person with whom you've discussed the assignment and describe the nature of your discussions.
    * Discuss any unresolved issues or problems. 
* A REPORT document that discusses the following:
    * Describe what you did for Step 2 and report the baseline performance and compare it against majority voting.
    * Describe your two models for Step 3 and report the performance of each. Compare these models against the previous baselines and each other, including your results from Step 4
    * Pose your question (Step 5). Provide some motivation or explanation for why you asked this question; or you may offer some hypotheses (what you think the outcome will be). Describe how you want to set up the experiment to answer your question. Present the experimental results. Discuss the outcomes and draw some conclusions. 
* Submit all of the above materials to CourseWeb. 
#### Grading Guideline
Assignments are graded qualitatively on a non-linear five point scale. Below is a rough guideline:
1. (40%): A serious attempt at the assignment. The README clearly describes the problems 
encountered in detail.
2. (60%): Correctly completed the assignment through Step 2, but encountered significant problems 
with later steps. Submitted a README documenting the problems and a REPORT for the outcomes of 
Step 2.
3. (80%): Correctly completed the assignment through Step 4, but has a significantly flawed Step 5. 
Submitted a README and a REPORT.
4. (93%): Correctly completed the assignment through Step 4. For step 5, the question posed is 
clear and rigorously answered through experimentation. The REPORT content is solid.
5. (100%): Correctly completed the assignment through Step 4. For step 5, the question posed is 
clear and interesting; it is rigorously answered through experimentation. The REPORT content is 
well-written and insightful. 
#### Acknowledgment
This assignment is adapted from Prof. Hwa 