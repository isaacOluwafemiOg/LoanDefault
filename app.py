import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pycaret.classification import predict_model


def main():

    
    st.sidebar.header('Dataset to use')
    page = st.sidebar.selectbox("Format", ['Default Test Data','User Upload'])
    model =load_model('LDefault.pkl')

    if page == 'Default Test Data':
        st.title('Predicting Default Test Data')
        st.subheader('Test Dataset Preview')
        test = pd.read_csv('test.csv')
        test

        prediction=predict_model(model,test)

        st.subheader('Results')
        prediction
        
        st.write('''***''')


    elif page == 'User Upload':
        st.title('Predicting on Uploaded Data')
        u_data = st.file_uploader('The loan applications on which you want to predict',type='csv')
        if u_data is not None:
            data = pd.read_csv(u_data,index_col = 'ID')

            st.subheader('Uploaded Dataset Preview')
            data

            prediction=predict_model(model,data)

            st.subheader('Results')
            prediction

        else:
            st.write('No dataset Uploaded')
        
@st.cache
def load_model(a):
    return pickle.load(open(a,'rb'))  



if __name__ == '__main__':
    main()
