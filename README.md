Demo server: https://aitomatic.ikuhi.com/
To run the code in local:streamlit run deploy.py
You can use sample csv files or same format file to test my model. 
Model is train base on data from original. I also use the preprocessed data provided by your company but i don't know how you guy generate label 
Notebook code try to run original data and your provided data 
This dataset doesn't have good signal to classication.
Because dataset is imblanced so i use f1 score as a metric. Train | test size is 69994(7947 positive), 13900(269 positive) 
Use date 2015-11-15 to split train, test 
Some technique use in this assigment:
+ EDA tofind important feature 
+ Up sampling (increase f1 by 15%) on training set 
+ LSTM to handle time series, take 1 day (24 hours) as an input, output binary classification 
+ Feature engineering: rank, filter bank (mel spectrogram), delta doesn't increase the performane 
+ To handle category features (machineID, age, model) by using it mean, std for normal numeric features ('volt','rotate','pressure','vibration')
+ Remove error feature because almost sample which has error doesn't fail 
Best f1: 36.6% 
Because the dataset is not big so transformer performs very poor 