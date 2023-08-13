import streamlit as st
import lstm
import pandas as pd
import io

st.title('Failures Classification')
#load model 

fileTypes = ["csv"]
uploaded_file = st.file_uploader("Choose a csv file", type=fileTypes)
show_file = st.empty()
if uploaded_file is not None:
    # To read file as bytes:
    content = uploaded_file.getvalue().decode('utf-8')
    # print(content)
    df = pd.read_csv(io.StringIO(content), sep=",")
    st.write('### Source csv')
    st.table(df)
    clicked = st.button('Classification')
    if clicked:
        model = lstm.load_model()
        predict = lstm.predict(model, df)
        st.write('### Result:')
        st.text(predict)

