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
1. [Support Vector Machine  (SVM)](#support-vector-machine)
2. [Recurrent Neural Network (RNN)](#rnn (Jay))
3. [Convolution Neural Network (CNN)](#cnn (Myles))
4. [Fully-Connected Neural Network](#fully-connected-nn)
5. [Bidirectional Encoder Representations from Transformers (BERT)](#bert)

---
## Support Vector Machine
### Overview
The model used Bag of Words pre-processing step to convert text to tokenized form. Then trained SVM model, conducted hyperparameter tuning to choose best model based on evaluation splits: 5-fold CV error. Final model scores as follows, loss metric was accuracy:

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

### Analysis
| Metric                        | Value               |
|-------------------------------|---------------------|
| **Optimal Hyperparameters**    | `{'C': 5, 'kernel': 'rbf'}` |
| **Evaluation (5-fold CV) Accuracy** | 0.572             |
| **Training Accuracy**          | 0.993               |
| **Testing Accuracy**           | 0.555               |

The training accuracy is extremely high, which means that the model is almost certainly not underfitted. Looking at the evaluation accuracy and the testing accuracy, it can be seen that since they are extremely close, that the model was properly fit. In other words, the 5-fold cross validation accuracy is meant to represent what the testing accuracy would be. Since it is so close to the actual testing accuracy, this indicates that the model can’t be fitted any further.

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

---

![image](https://github.com/user-attachments/assets/562b6347-a542-456d-b565-e9219b4baf79)

## CNN (Myles)
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


## Fully-Connected NN

**Overview:**
Performed text cleaning and tokenization using Python’s Natural Language Toolkit, then used Word2Vec to create feature vectors representing the data. After preprocessing was completed, fully connected neural network with 3 layers was trained using 1000 epochs. Performed 5-fold cross validation. Performed final test on test split of data.


Training Accuracy: 0.57 <br>
Evaluation (5 fold CV) Accuracy: 0.4 <br>
Testing Accuracy: 0.41 <br>

**Analysis:**
Low accuracy during training indicates that the model is underfitting. Given enough training data and computational resources, it is possible that a simple fully connected NN may be capable of fitting data with higher accuracy, but given the fact that 1000 epochs were already used it would likely take a very high amount of epochs to produce an accurate result. In addition, accurate results on training & testing given an even higher amount of epochs would likely be overfit. Therefore, it is reasonable to conclude that a 3 layer perceptron is not the best choice of model for this task.


---------------------------------------------------------------------------------------------------------------------



Bert

Bert stands for Bidirectional Encoder Representations from Transformers, and is used in a wide range of natural language processing tasks. Bert is bidirectional, meaning it reads text from left to right simultaneously. It is trained to predict a masked word or sentence from a given block of text. This training allows Bert to find connections and establish context in a given block of text. This is useful for text-classification, which is our goal for this project. Bert itself doesn’t process raw text, the input must first be tokenized. This is simply the process of transforming the text into numbers that Bert can interpret. Bert expects an id, attention mask, and token type ids. The id is the numerical representation of a word or subword, the attention mask represents which ids are from an input and which are padding, and the token type ids are a mask to differentiate between different sequences in an input, such as a question and an answer. 

Throughout the development of this project, many different layers were connected to the output of Bert in order to act as classification layers. These layers take the context learned by Bert, and are trained to correctly classify the score of an essay. The results from a single linear layer, trained for 500 epochs with a cross entropy loss function, is shown below. 

Statistics for Testing

Cross entropy loss: 1.0282892952101272

Correct accuracy (%): 56.390977443609025

Average absolute error:  0.4912280738353729

Classification report:

              precision    recall  f1-score   support

           0       0.52      0.18      0.27       124

           1       0.58      0.66      0.62       399

           2       0.60      0.58      0.59       600

           3       0.51      0.71      0.59       363

           4       0.46      0.13      0.20        94

           5       0.00      0.00      0.00        16

    accuracy                           0.56      1596

   macro avg       0.45      0.37      0.38      1596

weighted avg       0.56      0.56      0.54      1596
<img width="677" alt="Screenshot 2024-11-27 at 4 13 36 PM" src="https://github.com/user-attachments/assets/5ab05050-8d21-41f1-ba82-2ca17de5f2cc">

The green line depicts the absolute mean error for the testing set throughout training, while the other two relations depict the cross entropy loss for training and testing. 
<img width="688" alt="Screenshot 2024-11-27 at 4 13 46 PM" src="https://github.com/user-attachments/assets/e3d5049f-ca32-4d91-8765-d96a49985f97">


Similar results were obtained when using combinations of LSTM layers, sigmoid layers, linear layers. Notably, the model has a much higher accuracy rate for more common classifications, while it does much worse of classifications that aren’t as common in the training data. This can be seen in the confusion matrix for testing, which shows the model did not predict a single 5 score essay correctly. This distribution of scores for the entire dataset is shown below. 
<img width="621" alt="Screenshot 2024-11-27 at 4 13 55 PM" src="https://github.com/user-attachments/assets/fb72b116-d6b3-4bee-855d-2923d161ec72">


In order to help improve the models performance on less frequent classifications, I tried changing the loss function. I created a vector of weights, where each weight corresponds to a class. The weight for each class is computed as Wi = NN Ci, where N is the number of samples, and Ci is the number of samples with target classification i. This gives a larger weight to classifications that occur less frequently. The following results were obtained from using this new loss function. 

Statistics for Testing

Cross entropy loss: 1.2106342015689926

Correct accuracy (%): 46.99248120300752

Average absolute error:  0.6936089992523193

Classification report:

              precision    recall  f1-score   support

           0       0.30      0.64      0.41       124

           1       0.60      0.48      0.53       399

           2       0.66      0.40      0.50       600

           3       0.47      0.54      0.50       363

           4       0.23      0.37      0.29        94

           5       0.09      0.50      0.16        16

    accuracy                           0.47      1596

   macro avg       0.39      0.49      0.40      1596

weighted avg       0.54      0.47      0.49      1596
<img width="653" alt="Screenshot 2024-11-27 at 4 14 05 PM" src="https://github.com/user-attachments/assets/c22468b0-6754-4578-a2a0-e00c23dadecc">


The new loss function raises the probability the model correctly predicts higher scores. However, it now predicts these classes at a very high rate, which is shown from its precision score of .09 for score 5. This means only 9% of all score 5 predictions are actually score 5. For this reason, the total accuracy of the model is significantly less than the one trained with a non-weighted cross entropy loss function.



