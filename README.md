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

---

## Linear Regression

## Support Vector Machine

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

## CNN
Cross Entropy: <br>
Training: 0.89 <br>
Testing: 1.05 <br> 
Validation: 1.01 <br>

Accuracy: <br>
Training: 64% <br>
Testing: 52.88% <br>
Validation: 59% <br>

## BERT

Cross Entropy: <br>
Training: .89<br>
Testing: 1.05<br>
Validation: 1.01<br>

Accuracy: <br>
Training: 64% <br>
Testing: 57%<br>
Validation: 59% <br>

Overview: used Bag of Words pre-processing step to convert text to tokenized form. Then trained SVM model, conducted hyperparameter tuning to choose best model based on evaluation splits: 5-fold CV error. Final model scores as follows, loss metric was accuracy:


Training Accuracy: 0.992
Evaluation (5-fold CV) Accuracy: 0.571
Testing Accuracy: 0.554


Analysis
The training accuracy is extremely high, which means that the model is almost certainly not underfitted. Looking at the evaluation accuracy and the testing accuracy, it can be seen that since they are extremely close, that the model was properly fit. In other words, the 5-fold cross validation accuracy is meant to represent what the testing accuracy would be. Since it is so close to the actual testing accuracy, this indicates that the model can’t be fitted any further.
