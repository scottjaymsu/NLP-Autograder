# Natural Language Processing (NLP) Essay Autograder
Contributers: Jay Scott, Archan Tulpule, Myles Yankie, Nick Clark, Michael Plante

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
---
### RNN Analysis 
From observing the resulting evaluation metrics for each class, it was determined that the accuracy is most negatively impacted by scoring classes (5 & 6). This was due to a combination of our training/testing splits, as well as the data used to train the model. The dataset used did not have a large quantity essay representations for higher grading criteria; therefore, this negatively impacted the model's ability to train to these ranges. 

----------------------------------------------------------------------------------------------------------------------

## CNN (Myles) <br>
A Convolutional Neural Network (CNN) was implemented to perform multi-class classification on textual data sourced from the train.csv dataset, which comprises 8,510 samples distributed across six classes. The preprocessing pipeline involved tokenizing the text data using the Tokenizer from Keras, converting texts into sequences of integers, and padding these sequences to ensure uniform input length of 500 tokens. The model architecture consists of embedding, convolutional, pooling, dense, and dropout layers designed to capture and generalize complex patterns within the data. The model was trained using an 80-10-10 split for training, validation, and testing, respectively, with early stopping employed to prevent overfitting. The primary performance metric used for evaluation was accuracy, supplemented by precision, recall, F1-score, and Quadratic Weighted Kappa (QWK) to provide a comprehensive assessment of the model's performance across different classes.

![image](https://github.com/user-attachments/assets/f032d031-fcba-451c-afc2-112ad491ee2d)

The dataset comprises a total of 8,510 samples distributed across six classes as follows:

The dataset exhibits significant class imbalance, with classes 3.0, 2.0, and 4.0 being the most prevalent, while classes 1.0, 5.0, and especially 6.0 are underrepresented. This imbalance poses challenges for the model, as it may lead to biased learning favoring majority classes, thereby adversely affecting the performance on minority classes.

**Training Accuracy**
Value: 0.9337

The training accuracy of 0.9337 indicates that the CNN model performs exceptionally well on the training data, correctly classifying approximately 93.37% of the samples it was trained on. This high accuracy suggests that the model has effectively learned the underlying patterns and relationships within the training dataset.

Model Learning: The high training accuracy confirms that the model has successfully captured and memorized the features present in the training data.
Risk of Overfitting: While high training accuracy is desirable, it raises concerns about potential overfitting, especially when there is a notable discrepancy between training and validation/testing accuracies. In this case, the validation and testing accuracies are significantly lower, suggesting that the model may not generalize well to unseen data.

**Validation Accuracy**
Value: 0.5159

The validation accuracy of 0.5159 reflects the model's performance on a subset of data that was not used during training. This metric serves as an indicator of the model's ability to generalize its learning to new, unseen data.

Generalization Check: A validation accuracy slightly above random guessing (for a six-class classification problem, random guessing would yield approximately 16.67% accuracy) indicates that the model has learned some meaningful patterns from the data. However, the performance is modest, suggesting room for improvement in generalization.
Overfitting Detection: The substantial gap between the high training accuracy (0.9337) and the lower validation accuracy (0.5159) points towards overfitting. The model performs well on the training data but fails to maintain similar performance on validation data, indicating that it may have memorized the training samples rather than learning to generalize.
Hyperparameter Tuning: This discrepancy underscores the necessity for further hyperparameter tuning, regularization techniques, or data augmentation strategies to enhance the model's ability to generalize.

**Testing Accuracy**
Value: 0.5429

The testing accuracy of 0.5429 measures the model's performance on an entirely unseen dataset, providing an unbiased evaluation of its real-world applicability.

Model Deployment Readiness: A testing accuracy of approximately 54.29% indicates that the model has moderate effectiveness in practical scenarios. While it performs better than random guessing, the accuracy suggests that the model may require further refinement for reliable deployment.
Consistency with Validation: The close alignment between validation accuracy (0.5159) and testing accuracy (0.5429) suggests that the validation set was a reliable proxy for the test set during training. This consistency reinforces the validity of the observed performance metrics.

![image](https://github.com/user-attachments/assets/972d0990-537d-4134-b9ea-c03e56c735ff)

Class Imbalance Impact: The classification report highlights the detrimental impact of class imbalance on model performance. Classes 4.0 and 5.0, which have the fewest samples (47 and 8 respectively), exhibit zero precision, recall, and F1-scores. This indicates that the model fails to correctly predict these minority classes.
Performance by Class:
Classes 1.0 and 2.0: These classes have higher support and demonstrate moderate precision and recall, suggesting that the model can somewhat reliably predict them.
Classes 0.0 and 3.0: These classes have lower support and exhibit poor performance metrics, indicating challenges in accurately classifying these categories.
Accuracy: The overall accuracy of 54% reflects the model's ability to correctly classify samples across all classes.
Macro Average: The macro-averaged precision, recall, and F1-score (0.33, 0.31, 0.29 respectively) provide an unweighted mean of the metrics across all classes, highlighting the poor performance on minority classes.
Weighted Average: The weighted averages (0.50, 0.54, 0.51) take into account the number of samples in each class, offering a more balanced view of performance but still indicating room for improvement.

![image](https://github.com/user-attachments/assets/80d56b2f-6b90-4a83-845d-47f701459619)

The confusion matrix visualizes the model's performance across different classes by displaying the number of correct and incorrect predictions for each class. It reveals that the model struggles particularly with classes 0.0, 4.0, and 5.0, where predictions are either inaccurately assigned to other classes or not predicted at all. The high number of misclassifications in these classes underscores the challenges posed by class imbalance and the model's limited ability to generalize effectively to less represented categories.

![image](https://github.com/user-attachments/assets/a05fc47e-42a0-48d0-bf47-e3ff3b9e1a67)

The Receiver Operating Characteristic (ROC) curves for each class illustrate the trade-off between true positive rates and false positive rates. The Area Under the Curve (AUC) values provide a measure of the model's ability to distinguish between classes. In this scenario, the ROC curves indicate that the model has varying degrees of discriminatory power across different classes. Classes with higher AUC values demonstrate better performance in distinguishing between positive and negative instances, while lower AUC values suggest poor discrimination. The overall moderate AUC values reflect the model's limited effectiveness in differentiating between classes, particularly those with fewer samples.

![image](https://github.com/user-attachments/assets/79149cd0-1918-4027-a14f-aef636b412fe)

The CNN model comprises the following layers:

Embedding Layer: Transforms input tokens into dense vectors of size 128, facilitating the capture of semantic relationships between words.
Conv1D Layer: Applies 128 filters with a kernel size of 5 to extract local features from the input sequences.
Global Max Pooling Layer: Reduces the dimensionality by selecting the maximum value from each feature map, effectively capturing the most salient features.
Dense Layer: Contains 64 units with ReLU activation, enabling the model to learn complex, non-linear combinations of the extracted features.
Dropout Layer: Implements a dropout rate of 0.5 to mitigate overfitting by randomly deactivating half of the neurons during training.
Output Layer: A Dense layer with 6 units and softmax activation, facilitating multi-class classification across the six target classes.
The model consists of approximately 2.89 million trainable parameters, enabling it to capture intricate patterns within the data. However, the substantial number of parameters also increases the risk of overfitting, especially given the observed performance metrics.

![image](https://github.com/user-attachments/assets/5cde97d0-96c1-4080-87a6-863c69c65c40)

Training Accuracy: The training accuracy shows a steady and significant increase across epochs, rising from 29.87% in the first epoch to 93.37% in the tenth epoch. This trend indicates that the model is effectively learning and fitting the training data over time.
Validation Accuracy: In contrast, the validation accuracy exhibits minor fluctuations, increasing initially but then plateauing and even slightly decreasing in later epochs. This stagnation suggests that the model's ability to generalize to unseen data is not improving in tandem with its performance on the training data.
Loss:

Training Loss: There is a consistent decrease in training loss from 1.6209 in the first epoch to 0.2570 in the tenth epoch, indicating that the model is minimizing its error on the training data effectively.
Validation Loss: The validation loss initially decreases, reaching its lowest point at epoch 7 (1.1345), but then begins to increase again. This upward trend in validation loss, coupled with the plateauing of validation accuracy, reinforces the indication of overfitting—the model becomes too tailored to the training data and loses its ability to perform well on new, unseen data.
Conclusion
The CNN model demonstrates a strong capacity to learn from the training data, as evidenced by the high training accuracy of 93.37% and the substantial reduction in training loss over the epochs. However, the validation accuracy of 51.59% and testing accuracy of 54.29% reveal that the model struggles to generalize effectively to unseen data. The significant disparity between training and validation/testing performances suggests that the model is overfitting, memorizing the training data rather than learning to generalize its patterns.

Class Imbalance: The pronounced imbalance in the dataset, with certain classes being underrepresented, adversely affects the model's ability to accurately predict minority classes. This imbalance is reflected in the classification report, where minority classes exhibit poor precision, recall, and F1-scores.
Overfitting Indicators: The high training accuracy coupled with modest validation and testing accuracies indicate that the model is overfitting to the training data. The increase in validation loss in later epochs further substantiates this concern.

---------------------------------------------------------------------------------------------------------------------


## Fully-Connected NN

**Overview:**
<br>
Fully-connected neural networks can be used to identify and learn complex decision spaces given that the data from which the network learns is properly represented. To perform essay scoring with a fully-connected neural network, 150-500 word essays were first tokenized utilizing a BERT tokenizer. Tokenized text was then passed to the "bert-base-uncased" pretrained model to generate embeddings for the text. From these embeddings, the most relevant for usage in the case of a fully-connected neural network is the CLS token, a vector which captures longer-term relationships between words within text. These tokens were then used as feature vectors on which a fully-connected, 3 layer neural network utilizing sigmoid activation functions was trained for 500 epochs, cross-validated to perform hyper-parameter tuning, and tested.

**Training:**
<br>
After generating feature vectors, the 3 layer FCNN model was trained with batch sizes of 64 on an 80% split of the data over 500 epochs. An Adam optimizer was used, with an initial learning rate of 0.0001. The hidden layer consisted of 128 units, and a dropout of 0.5 was used after the initial layer to attempt to reduce overfitting. 

Over the 500 epoch for which the model, accuracy increased logistically to ~65% and loss decreased logisitically, as can be seen in the below learning curves:
<img width="750" alt="image" src="https://github.com/user-attachments/assets/e73468e2-9aa8-408e-b46b-73768637d531">


Model performance per class can be observed in the form of a confusion table below (zero-indexed):
<img width="491" alt="image" src="https://github.com/user-attachments/assets/16dbc971-9784-422b-a721-ff558fca2a81">


It is apparent from this confusion matrix that after training the model classifies the majority of the training data correctly, and if an essay is scored incorrectly it is usually only incorrect by 1 unit of scoring. In other words, for the ~35% of the training data that the trained FCNN misclassifies, the majority of these misclassifications are by the smallest unit of misclassification possible. <br>

Below are the per class accuracies, precisions, recalls, and F1 scores:

| Class    | Accuracy | Precision | Recall   | F1 Score |
|----------|----------|-----------|----------|-------------|
| Class 0  | 0.943743 | 0.555985  | 0.919149 | 0.692863    |
| Class 1  | 0.836663 | 0.718514  | 0.663807 | 0.690078    |
| Class 2  | 0.752644 | 0.690523  | 0.569625 | 0.624275    |
| Class 3  | 0.825940 | 0.632293  | 0.591746 | 0.611348    |
| Class 4  | 0.938014 | 0.473397  | 0.906005 | 0.621864    |
| Class 5  | 0.997650 | 0.786667  | 1.000000 | 0.880597    |


As can be seen, per class accuracy is high after training, with scores (classes) falling in the middle of the distribution having the highest inaccuracies. This indicates that trained model is most accurate when identifying scores that are either very good or very poor, but has more trouble descerning between more average scores. It is worth noting that recall is higher than precision for scores, indicating that the models overclassify training data as one of these scores.

Ultimately, final accuracy on the training set was computed to be: 0.6515863689776733 <br>

**Validation:**
<br>
5-fold cross-validation was performed on models to ensure that the model parameters that best balances qualities such as high accuracy, low overfitting, low underfitting, and reasonable performance were found. 
<br>
<br>
<b>Some notes on hyper-parameter tuning:</b> 
<br>
    - Decreasing learning rate successfully completely eliminated overfitting, but also reduced overall accuracy to less than 50% and caused issues with performance <br>
    - Increasing number of hidden units improves accuracy but decreases performance significantly; however, decreasing from 256 to 128 hidden units had only margin effects on fitting and accuracy <br>
    - Decreases in batch size by a factor of 2 improved accuracy, but decreases in performance were so significant that such batch sizes were unreasonable to use <br>
    - Increasing epochs to beyond 500 can improve training performance significantly (up to ~80% overall accuracy), but these accuracy increases are the result of overfitting validation set accuracy remains in the 50-60% range <br>

The most reasonable hyper-parameters for our model were found to be 500 epochs on a model with 128 hidden units in the middle layer, using batch sizes of 64 and a learning rate of 0.0001.
<br>
Found below are the cross-validation results given these hyper-parameters: 
<br>
<img width="370" alt="image" src="https://github.com/user-attachments/assets/6bce1a8b-092d-4007-a8d6-441e7f3c580b">



**Testing:**
<br>
Testing was performed on a 20% of the original dataset using the hyper-parameters found most reasonable during the validation stage. 
<br>
The overall accuracy of the model on the testing data was: 0.5323149236192715
<br>
This accuracy is ~10% lower than that the model's accuracy on the testing data, indicating that some overfitting on the training data is still present with the given hyper-parameters. It is also possible that some patterns exist in the testing data could exist the training data, but given that this accuracy is similar to that of the validation sets during cross-validation it is likely that this is not the case. <br>

Model performance per class on the testing data can be observed in the form of a confusion table below (zero-indexed):
<img width="486" alt="image" src="https://github.com/user-attachments/assets/b55babf8-1634-429e-9fd0-a2442e32b588">


As with the training set, it is apparent that the largest single predicted value of any class is its actual label, however the aggregate of incorrect classifications significantly diminishes the accuracy. Additionally, as seen with the training data, the majority of misclassifications are by the smallest possible unit of misclassification.

Below are the per class accuracies, precisions, recalls, and F1 scores for the model on the testing data:

| Class    | Accuracy | Precision | Recall   | F1 Score |
|----------|----------|-----------|----------|-------------|
| Class 0  | 0.919506 | 0.482993  | 0.537879 | 0.508961    |
| Class 1  | 0.787309 | 0.573222  | 0.634259 | 0.602198    |
| Class 2  | 0.710928 | 0.646035  | 0.519440 | 0.575862    |
| Class 3  | 0.787897 | 0.534314  | 0.560411 | 0.547051    |
| Class 4  | 0.919506 | 0.328358  | 0.483516 | 0.391111    |
| Class 5  | 0.981786 | 0.055556  | 0.066667 | 0.060606    |


The above data indicates that, while some patterns in the testing data remain similar to the training data, there is a general decrease in the individual accuracies for each classification. This could be attributable to new patterns in the data that the model has not encountered before in training or due to overfitting on training data. It is notable that have been significant decreases in both precision and recall, and subsequently a decreases in F1 scores for all classes. However, the same pattern of recall remaining larger than precision on the low and high scores remains the same as in testing, indicating that the model still tends to score many essays with one of these scores. <br>

**Conclusion:**
<br>
Lower accuracy of validation and testing sets compared trained data suggests that the model has a tendency to overfit. Given enough training data, more computational resources, and more representative feature vectors, it is possible that a 3 layer FCNN could increase its accuracy on testing sets and begin to accurately score essays in much the same way as a human graded would. Given the current conditions however, a FCNN only seems to be adequately useful at identifying a range of 3/6 scores in which an essay should be graded.   


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



