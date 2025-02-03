# Resume Screening and Categorization Model 📃

## Overview

I initially designed this model for myself to identify job opportunities that best align with my skills and experience. However, I quickly realized its potential to benefit both HR teams and job seekers, streamlining the hiring process and improving job matching accuracy. This project isn't just about automation it's about empowering job seekers (like me 😄 ) and recruiters with data driven insights to make smarter career and hiring decisions. I plan to make this system free for job seekers because I know how stressful the job search process can be. I personally struggled to find free platforms that could make it easier, so I want to lend a helping hand to those in the same shoes as me.

## Features

* **Resume Parsing and Preprocessing:** Extracts text from PDF resumes, cleans the text, and removes irrelevant information like stop words.
* **Category Prediction:** Trains machine learning models (Logistic Regression, Random Forest, SVM, Naive Bayes) to classify resumes into different job categories.
* **Similarity Evaluation:** Assesses the relevance of a resume to a job description using TF-IDF and semantic similarity measures.
* **Key Role Extraction:** Identifies and ranks top job roles from a resume.
* **Experience Extraction:** Dynamically extracts the years of experience from a resume.

## Data

The system uses the "UpdatedResumeDataSet.csv" dataset for training and evaluation. This dataset contains resumes categorized into different job categories. you can find this data here: https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset 

## Implementation Details

1. **Data Loading and Preprocessing:** 
    - load the dataset.
    - Cleans the resume text by removing special characters, extra spaces, and converting to lowercase.
    - Processes the text using NLTK to remove stop words and irrelevant tokens.
2. **Feature Extraction:**
    - Uses TF-IDF vectorization to convert resume text into numerical features.
3. **Model Training and Evaluation:**
    - Splits the data into training and testing sets.
    - Trains multiple machine learning models using the training data.
    - Evaluates the models' performance using accuracy, classification reports, and confusion matrices.
4. **Category Prediction:**
    - Loads the trained model and vectorizer.
    - Extracts text from a new resume (PDF format).
    - Preprocesses and vectorizes the resume text.
    - Predicts the category of the resume using the trained model.
    ![image](https://github.com/user-attachments/assets/a1649728-bbab-4c85-b75d-c7024cdac2c4)
    ![image](https://github.com/user-attachments/assets/81b9bed1-59b1-4878-a36b-dc283d89bd43)

5. **Similarity Evaluation:**
    - Takes a job description and a resume as input.
    - Calculates TF-IDF similarity and semantic similarity between the two texts.
    - Combines the similarities to generate a final relevance score which is (simalirity_score).
6. **Key Role and Experience Extraction:**
    - Extracts top job roles from the resume using spaCy's named entity recognition and noun phrase chunking.
    - Extracts years of experience using regular expressions to find experience-related patterns in the text.

## Result 
**Job description and CV** 

  <img width="442" alt="image" src="https://github.com/user-attachments/assets/6ca44b2c-7d68-41bb-b4b6-813e13fa7b2f" />
  
**Model output** 

  <img width="346" alt="image" src="https://github.com/user-attachments/assets/4a469e41-30dc-43d4-be8a-460f76077c11" />

## Further Improvment

* **Improve accuracy:** The similarity evaluation can be enhanced by using a model instead of the current function (I used cosine similarity here). However, there isn't a suitable public dataset available to train the model.
* **Expand functionality:** it possible to expand the functionalit of the model and system with more complex dataset and feature engineering.
