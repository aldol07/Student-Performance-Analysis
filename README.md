# Student Performance Predictor

## Project Overview
This project aims to predict student test performance based on various factors such as demographics, parental education, and study habits. Using machine learning techniques, we analyze a dataset of student information to identify key predictors of academic success.

## Features
- Data preprocessing and exploratory data analysis (EDA)
- Machine learning model development using CatBoost
- Web application for real-time predictions
- Visualizations of key performance indicators

## Installation
git clone https://github.com/aldol07/Student-Performance-Analysis
cd Student-Performance-Analysis
pip install -r requirements.txt


## Usage
1. Run the Jupyter notebook for data analysis:
2. Start the web application:
3. Open your browser and navigate to `http://localhost:5000`

## Data
The dataset used in this project is sourced from [Kaggle's Student Performance Dataset](https://www.kaggle.com/spscientist/students-performance-in-exams). It includes information on:
- Student demographics
- Parental level of education
- Test preparation course completion
- Scores in math, reading, and writing

## Model
We use CatBoost, a gradient boosting library, to predict student performance. The model takes into account various features to estimate a student's likely test scores.

## Web Application
Our Flask-based web app allows users to input student information and receive predicted test scores in real-time.


