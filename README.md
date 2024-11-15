# Natural Language Processing (NLP) Essay Autograder

## Project Description
The Natural Language Processing (NLP) Essay Autograder is a system designed to automatically evaluate and score essays based on their quality. The input is an essay text, and the output is a numerical score ranging from 1 to 6, which reflects the essay's overall quality. The system uses NLP techniques to analyze various factors, such as grammar, structure, coherence, vocabulary, and content relevance, to generate an accurate and objective score. This tool can assist educators in providing quick, consistent feedback on student writing.

## Visualizing Data
![image](https://github.com/user-attachments/assets/2cf5fa69-3b59-4118-9179-1f382e8ec287)
---
![image](https://github.com/user-attachments/assets/37d8a6c8-3646-4cec-bb2b-2ac97d98c505)
---
![image](https://github.com/user-attachments/assets/39d47aee-d9de-4514-9c4e-0d18f56eddf9)

## Models
1. [Linear Regression](#linear-regression)
2. [Support Vector Machine  (SVM)](#support-vector-machine)
3. [Recurrent Neural Network (RNN)](#rnn)
4. [Convolution Neural Network (CNN)](#cnn)
5. [Bidirectional Encoder Representations from Transformers (BERT)](#bert)
6. [Fully-Connected Neural Network](#fully-connected-nn)

---

## Linear Regression

---
## Support Vector Machine
### Mean and Standard Deviation of Each Hyperparameter
The table below shows the results from the grid search for different combinations of hyperparameters (`C` and `kernel`) for the SVM model. The results include the mean and standard deviation of the training and test scores across the 5 folds of cross-validation.

| `mean_fit_time` | `std_fit_time` | `mean_score_time` | `std_score_time` | `param_C` | `param_kernel` | `params`                               | `split0_test_score` | `split1_test_score` | `split2_test_score` | `split3_test_score` | `split4_test_score` | `mean_test_score` | `std_test_score` | `rank_test_score` |
|-----------------|----------------|-------------------|------------------|-----------|----------------|----------------------------------------|---------------------|---------------------|---------------------|---------------------|---------------------|-------------------|------------------|-------------------|
| 21.92           | 0.27           | 4.35              | 0.05             | 1         | linear         | `{'C': 1, 'kernel': 'linear'}`         | 0.504195            | 0.544463            | 0.507137            | 0.502939            | 0.521411            | 0.516029          | 0.015671          | 6                 |
| 25.38           | 0.17           | 4.65              | 0.05             | 1         | poly           | `{'C': 1, 'kernel': 'poly'}`           | 0.478188            | 0.493289            | 0.492024            | 0.476071            | 0.485306            | 0.484975          | 0.006989          | 9                 |
| 20.31           | 0.13           | 5.63              | 0.02             | 1         | rbf            | `{'C': 1, 'kernel': 'rbf'}`            | 0.567114            | 0.582215            | 0.559194            | 0.563392            | 0.567590            | 0.567901          | 0.007768          | 2                 |
| 15.72           | 0.17           | 4.15              | 0.05             | 1         | sigmoid        | `{'C': 1, 'kernel': 'sigmoid'}`        | 0.470638            | 0.476510            | 0.450882            | 0.452561            | 0.460118            | 0.462142          | 0.010013          | 12                |
| 21.65           | 0.32           | 4.28              | 0.02             | 5         | linear         | `{'C': 5, 'kernel': 'linear'}`         | 0.504195            | 0.544463            | 0.507137            | 0.502939            | 0.521411            | 0.516029          | 0.015671          | 6                 |
| 27.14           | 0.37           | 4.64              | 0.04             | 5         | poly           | `{'C': 5, 'kernel': 'poly'}`           | 0.520973            | 0.531040            | 0.511335            | 0.522250            | 0.519731            | 0.521066          | 0.006283          | 5                 |
| 25.13           | 0.20           | 5.63              | 0.02             | 5         | rbf            | `{'C': 5, 'kernel': 'rbf'}`            | 0.565436            | 0.590604            | 0.575987            | 0.570109            | 0.555835            | 0.571594          | 0.011568          | 1                 |
| 12.09           | 0.35           | 3.63              | 0.05             | 5         | sigmoid        | `{'C': 5, 'kernel': 'sigmoid'}`        | 0.450503            | 0.497483            | 0.473552            | 0.451721            | 0.445844            | 0.463821          | 0.019364          | 11                |
| 21.78           | 0.20           | 4.31              | 0.04             | 10        | linear         | `{'C': 10, 'kernel': 'linear'}`        | 0.504195            | 0.544463            | 0.507137            | 0.502939            | 0.521411            | 0.516029          | 0.015671          | 6                 |
| 28.06           | 1.64           | 4.61              | 0.02             | 10        | poly           | `{'C': 10, 'kernel': 'poly'}`          | 0.539430            | 0.534396            | 0.525609            | 0.532326            | 0.520571            | 0.530466          | 0.006645          | 4                 |
| 25.38           | 0.12           | 5.67              | 0.02             | 10        | rbf            | `{'C': 10, 'kernel': 'rbf'}`           | 0.554530            | 0.588926            | 0.568430            | 0.568430            | 0.553317            | 0.566727          | 0.012862          | 3                 |
| 11.07           | 0.10           | 3.45              | 0.04             | 10        | sigmoid        | `{'C': 10, 'kernel': 'sigmoid'}`       | 0.447987            | 0.489094            | 0.465155            | 0.459278            | 0.466835            | 0.465670          | 0.013445          | 10                |

#### Optimal Hyperameters : {'C': 5, 'kernel': 'rbf'}
#### 5-fold Cross-Validation Score : 5-fold CV error: 0.5715942837500493
#### Test Set Accuracy : 0.5462382445141066
#### Final training error : 0.9926137317441666
#### Final test score : 0.554858934169279
---
## RNN
### Initial RNN Model Summary
![{F624C53E-2B0F-435B-A109-4EB8CD6FD366}](https://github.com/user-attachments/assets/4a4f9778-acad-4455-baf8-fc6acc3b401a)
---
![{98DEBB03-B69E-4908-ACFD-99EA2738AEC0}](https://github.com/user-attachments/assets/5814c0a6-ab1e-41a5-96a8-919206b45184)
---
#### As displayed above, the model's training accuracy is improving signifcantly, but the validation accuracy is lagging, which could indicate overfitting. Therefore, further iterative improvements are included below to address this.
![{C2797F21-5ADD-401F-B765-6014D1840562}](https://github.com/user-attachments/assets/13ca025e-9b2e-489f-a850-9b162c982235)
---
- Precision (Macro): 0.29573581508760566
- Recall (Macro): 0.2573651421894449
- F1 Score (Macro): 0.26400296809006835
- Precision (Micro): 0.38554645781416874
- Recall (Micro): 0.38554645781416874
- F1 Score (Micro): 0.38554645781416874
- Precision (Weighted): 0.3730191565916721
- Recall (Weighted): 0.38554645781416874
- F1 Score (Weighted): 0.37703300183006944
- Overall Accuracy: 0.38554645781416874
- Quadratic Weighted Kappa (QWK): 0.41340393076209814
---
![{9E3E9FC3-E9F3-4D57-BC15-FCA4FB576FF3}](https://github.com/user-attachments/assets/1adab5a1-707f-4e9b-ad94-f077accf3287)
---
![{6DD5BAC0-FB12-48F1-9362-B8935BAB3326}](https://github.com/user-attachments/assets/a94e066d-6f1a-4be0-b021-2024b2f713ea)

--------------------------------------------------------------------------------------------------------------------------------

![image](https://github.com/user-attachments/assets/562b6347-a542-456d-b565-e9219b4baf79)

**CNN (Myles)** <br>
Training Accuracy: 0.985 <br>
Evaluation (5-fold CV) Accuracy: 0.573 <br>
Testing Accuracy: 0.5288 <br>

![image](https://github.com/user-attachments/assets/ae4b1ce4-94a4-4697-b89d-dceac526808d)

**Overview:**
The Bag of Words pre-processing technique is used to convert text into tokenized form before training the SVM model. Hyperparameter tuning was conducted to select the best model configuration using a 5-fold cross-validation approach. The final scores, evaluated with accuracy as the performance metric:

![image](https://github.com/user-attachments/assets/22d1afa4-ba49-4780-b091-81714973e4d6)

**Training Accuracy:**
This metric reflects the model's performance on the data it was trained on. A high training accuracy suggests that the model is able to learn and correctly classify most of the patterns in the training dataset.
Indicator of Model Learning: If training accuracy is too low, it means the model is not capturing patterns effectively, which may indicate underfitting.
Risk of Overfitting: If training accuracy is excessively high compared to evaluation and testing accuracy, it may mean the model has memorized the training data instead of generalizing, leading to poor real-world performance.
The training accuracy is 0.985, which is very high. This suggests that the model has learned well from the training data, but it is important to compare this with evaluation and testing accuracy to ensure it is not overfitting.

**Evaluation Accuracy (5-Fold Cross-Validation):**
Evaluation accuracy is calculated using a validation dataset or through cross-validation techniques. In a 5-fold CV, the dataset is split into five subsets, and the model is trained on four subsets while the remaining subset is used for validation. This process is repeated five times, and the average accuracy is reported.
Generalization Check: Evaluation accuracy gives an estimate of how well the model is expected to perform on unseen data. It acts as a proxy for the test set during training.
Overfitting Detection: A large gap between training and evaluation accuracy indicates overfitting, where the model performs well on training data but struggles on validation data.
Hyperparameter Tuning: During model optimization, evaluation accuracy helps determine the best set of hyperparameters without leaking information from the test set.
Evaluation accuracy is 0.573, which is much lower than the training accuracy. This suggests the model generalizes less effectively than it performs on the training data, it does not necessarily indicate overfitting as long as evaluation and testing accuracies are aligned.

**Testing Accuracy:**
Testing accuracy is the final metric that measures how well the model performs on completely unseen data. It represents real-world performance since the test dataset is not used during model training or evaluation. Model Deployment Readiness: Testing accuracy reflects how the model is expected to perform in practical scenarios.

The model demonstrates strong learning ability with a high training accuracy of 0.985, indicating it effectively captures patterns in the training data. However, the evaluation accuracy of 0.573 and testing accuracy of 0.550 suggest moderate generalization to unseen data. The close alignment between evaluation and testing accuracies confirms that the cross-validation process reliably estimated the model's real-world performance. While there is no significant overfitting or underfitting, the noticeable gap between training and evaluation/testing accuracies suggests the model may overfit slightly to the training data. To improve, the model could benefit from regularization, refined feature engineering, or more diverse training data to better capture broader patterns. Despite this, the alignment of evaluation and testing scores provides confidence that the model is well-calibrated for deployment. Further optimization could enhance performance, but the current metrics indicate a functional and reliable model for practical use.

---------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------

## Fully-Connected NN

**Overview:**
Performed text cleaning and tokenization using Python’s Natural Language Toolkit, then used Word2Vec to create feature vectors representing the data. After preprocessing was completed, fully connected neural network with 3 layers was trained using 1000 epochs. Performed 5-fold cross validation. Performed final test on test split of data.


Training Accuracy: 0.57 <br>
Evaluation (5 fold CV) Accuracy: 0.4 <br>
Testing Accuracy: 0.41 <br>

**Analysis:**
Low accuracy during training indicates that the model is underfitting. Given enough training data and computational resources, it is possible that a simple fully connected NN may be capable of fitting data with higher accuracy, but given the fact that 1000 epochs were already used it would likely take a very high amount of epochs to produce an accurate result. In addition, accurate results on training & testing given an even higher amount of epochs would likely be overfit. Therefore, it is reasonable to conclude that a 3 layer perceptron is not the best choice of model for this task.


---------------------------------------------------------------------------------------------------------------------



## BERT

Cross Entropy: <br>
Training: .89<br>
Testing: 1.05<br>
Validation: 1.01<br>

Accuracy: <br>
Training: 64% <br>
Testing: 57%<br>
Validation: 59% <br>

---------------

SVM

Overview: used Bag of Words pre-processing step to convert text to tokenized form. Then trained SVM model, conducted hyperparameter tuning to choose best model based on evaluation splits: 5-fold CV error. Final model scores as follows, loss metric was accuracy:


Training Accuracy: 0.992
Evaluation (5-fold CV) Accuracy: 0.571
Testing Accuracy: 0.554


Analysis
The training accuracy is extremely high, which means that the model is almost certainly not underfitted. Looking at the evaluation accuracy and the testing accuracy, it can be seen that since they are extremely close, that the model was properly fit. In other words, the 5-fold cross validation accuracy is meant to represent what the testing accuracy would be. Since it is so close to the actual testing accuracy, this indicates that the model can’t be fitted any further.
