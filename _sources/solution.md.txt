# Solution final

## Phase 1
In file phase1_all.py is the source code for our solution of phase 1 part.
By running this file we get the prediction file in json format for unsupervised data.

Our pipeline consists of :

1. preprocessing data by smoothing attributes

2. we use k means clustering for detecting states and assigning the label to each data point. we also checked how clustering works by using PCA

![img](https://lh5.googleusercontent.com/_19GPD0YlDJ5O9Dc447Al4F6RaG4kgBVRsuojwRVnAriOaW7lKdPkE1v3AWikzwksR6dNIXgJrIKBlcJhUwPl6LNm8bUNJRMKAJEcGTVAz25AtKuFDE5LlFvv0z0ORvdyfLvn8rACkPU)

3. if we only detect one state we compare with statistics of all weeks and based on that we have derived a threshold based on average pressure that we use to assign a state.

![img](https://lh3.googleusercontent.com/xCqWWWW3JGS_UuUK8bzIu7UPjVOhE3KhSzRMHN_qolAk2lO_PC1u6NFPoKttEouNO791usfBiIcNF2xeT2dcrDdkF5MRckYzST2DiXgMky8yxFkUpnn8frFSLa6ca9r4RL08kErIvpYf)

4. in the end we generate json file with solutions

![img](https://lh5.googleusercontent.com/PkBq1Qf4ImggnH6BJTYfWMcVPz7c8a1WTTC3l5Cgd8lwxL0nRrtOwpHAyOzCBdT0USjBsK_VfGYeC7fWkpfPbHrEvPrpVcJNz0RnGcJfP2xrYB_mTvU3JoEsjH5t48vZeEU_-9e6ViGu)

## Phase 2
In phase two we concatenate the data for the past 10 data points and train the classifier on a subsample of all data (cca 10 percent). As the classifier we used the Logistic regression, the Dummy classifier and the Random forest classification. For the first ten predictions we always predict False as we do not have enough data to make predictions. We got the best results with the Dummy classifier.



![img](https://lh5.googleusercontent.com/4xpuzU6dUPwMW0sUAER3LblyndhSogL5SOie0w5oHdID2-vrrBHNXWTVyZtcU_EDSCD5WCjSqys6cDhfxB1CxSlXrEAHM4ZzQSMrJ4F4Ofb9fiHk0Ynf2p8FgxiulUDTfnUzrtVCQSp4)