# Gender_Classification
![image](https://github.com/EVA-12042002/Gender_Classification/assets/129527829/17f8fc0d-2f23-4b57-b4a0-5bf94efad04a)

Overview
This project focuses on gender classification using machine learning techniques. Gender classification is a common task in various applications, such as voice assistants, social media analytics, and demographic studies. In this project, we demonstrate how to build and train a gender classification model based on given input features.

Train Dataset 
Number of Images = 12196 (70%)
Image Size = 224 X 224 X 3

Test Dataset
Number of Images = 5227 (30%)
Image Size = 224 X 224 X 3

Approach to the Problem

1]Changing the Number of Hidden layers in FC layers.

2]Changing Number of Neurons in Hidden layers.

3]Changing the type of Optimizer used for training.

4]Checking out the effect of different loss functions on the result.

5]Changing the activation function

6]Changing the number of Epochs

7]Apply early stopping

8]Add dropout layer(s)

9]Implement gradient clipping using the Adam optimizer

10]Add batch normalization layer(s)

11]Perform image augmentation techniques like rotation, flipping, adding random noise, etc.


Contents
1]gender_classification.ipynb: Jupyter Notebook containing the code for gender classification.

2]data.csv: The dataset used for training and testing the gender classification model.

3]gender.jpg: An image related to gender symbols, used for illustration purposes in this README.

Dependencies
To run the code in this repository, you will need the following Python libraries:

*Pandas

*NumPy

*Scikit-learn

*Matplotlib (for visualization)

*Tensorflow

*Keras

Usage

1]Open and run the Google Collab gender_classification.ipynb to explore the code, data preprocessing, model training, and gender classification.

2]The key steps involved in this project include:

Data Preprocessing: Loading and preprocessing the dataset, including feature extraction and labeling.

Model Building: Building a gender classification model using machine learning algorithms (e.g., decision trees, logistic regression, support vector machines, etc.).

Model Training: Training the model on the dataset to learn the relationship between input features and gender labels.

Model Evaluation: Evaluating the model's performance using relevant metrics such as accuracy, precision, recall, and F1-score.

Customize the code, experiment with different models, or use your own dataset for gender classification tasks.

Results
The results of the gender classification model are summarized as follows:

The gender classification model has been trained and tested, and it successfully predicts the gender of the input data as "female".

License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
If you have any questions or suggestions, feel free to contact me at [evangelinpriyanka12@gmail.com].

