import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score,accuracy_score
import sklearn
import pickle


@st.cache_data
def train_model():
    test = pd.read_csv('test.csv')
    data = pd.read_csv('train.csv')
    model =pickle.load(open('loan_trained.pkl','rb'))
    
    X = data.drop('Status',axis=1)
    y = data['Status']
    X_train,X_test,y_train,y_test= sklearn.model_selection.train_test_split(X,
    y,test_size=0.3,random_state=2)
    tr_pred = model.predict(X_train)
    te_pred = model.predict(X_test)
    train_score = accuracy_score(y_train,tr_pred)
    test_score = accuracy_score(y_test,te_pred)
    return (X_test,test,model,train_score,test_score)

@st.cache_data
def predict_test(pdata):
    model =pickle.load(open('loan_trained.pkl','rb'))
    pred = model.predict(pdata)
    
    return (pred)
    
    
@st.cache_data
def prepare(test):
    preprocessor = pickle.load(open('process.pkl','rb'))
    X = test.drop('Status',axis=1)
    data = pd.DataFrame(preprocessor.transform(X))
    
    return (data)

def main():
    
    st.sidebar.header('Dataset to use')
    page = st.sidebar.selectbox("Data Input", ['Preloaded','User Upload'])
    
    
    
    st.title('Loan Default Predictor')
    
    with st.spinner("Unpacking the model... Please wait."):
        X_test,test,model,trainscore,testscore = train_model()
    st.write('train accuracy score:', trainscore)
    st.write('test accuracy score:', testscore)

    if page == 'Preloaded':
        st.header('Predicting on preloaded Test Data')
        st.subheader('Dataset Preview')
        
        for_view = test.drop('Status',axis=1)
        for_view

        if st.button("Predict on test data above"):
            with st.spinner("Preparing data... Please wait."):
                pdata = prepare(test)
                
            with st.spinner("Predicting... Please wait."):
                predic = predict_test(pdata)

            st.subheader('Results')
            pred_view = pd.DataFrame(predic)
            
            pred_view.columns = ['Loan Default']
            
            pred_view['Prediction'] = np.where(pred_view['Prediction']==0,'Not Likey','Potential')
            
            pred_view
            
            #f1= f1_score(test['is_fraud'],predic)
            #st.subheader('F1 Score')
            #f1
        st.write('''***''')


    elif page == 'User Upload':
        st.header('Predicting on Uploaded Data')
        u_data = st.file_uploader('Transactions to be processed go here',type='csv')
        if u_data is not None:
            data = pd.read_csv(u_data,index_col = 'ID')

            st.subheader('Dataset Preview')
            data

 

        else:
            st.write('No dataset Uploaded')
    
        





if __name__ == '__main__':
    main()
