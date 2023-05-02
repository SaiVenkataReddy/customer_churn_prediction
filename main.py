#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
# from PIL import Image
#
# #load the model from disk
# import joblib
# model = joblib.load(r"./notebook/model.sav")
# Load the model using pickle
with open('customer_churn_prediction_model.pkl', 'rb') as file:
    model = pickle.load(file)
#Import python scripts
# from preprocessing import preprocess
# from sklearn.preprocessing import LabelEncoder
#
#
# def preprocess(data, experiment_name):
#     # Handle missing values
#     data = data.fillna(0)
#
#     # Encode categorical features
#     categorical_cols = data.select_dtypes(include='object').columns
#     for col in categorical_cols:
#         le = LabelEncoder()
#         data[col] = le.fit_transform(data[col])
#
#     # Scale numerical features
#     numeric_cols = data.select_dtypes(include=['int', 'float']).columns
#     for col in numeric_cols:
#         data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
#
#     # Perform experiment-specific preprocessing
#     if experiment_name == 'Online':
#         # Add additional preprocessing steps for the 'Online' experiment
#         pass
#     elif experiment_name == 'Offline':
#         # Add additional preprocessing steps for the 'Offline' experiment
#         pass
#
#     return data

def main():
    #Setting Application title
    st.title('Telco Customer Churn Prediction App')

      #Setting Application description
    st.markdown("""
     :dart:  This Streamlit app is made to predict customer churn in a ficitional telecommunication use case.
    The application is functional for both online prediction and batch data prediction. n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    #Setting Application sidebar default
    # image = Image.open('App.jpg')
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Customer Churn')
    # st.sidebar.image(image)

    if add_selectbox == "Online":
        st.info("Input data below")
        #Based on our optimal features selection
        st.subheader("Demographic data")
        gender = st.radio(
            "Gender 👉",
            options=["Male", "Female"],
        )
        seniorcitizen = st.radio(
            "Senior Citizen 👉",
            options=["Yes", "No"],
        )
        partner = st.radio(
            "Have Partner 👉",
            options=["Yes", "No"],
        )
        dependents = st.radio(
            "Dependents 👉",
            options=["Yes", "No"],
        )
        st.subheader("Payment data")
        tenure = st.slider('Number of months the customer has stayed with the company', min_value=0, max_value=72, value=0)
        contract = st.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
        paperlessbilling = st.radio(
            "Paperless Billing 👉",
            options=["Yes", "No"],
        )
        PaymentMethod = st.selectbox('PaymentMethod',('Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'))
        monthlycharges = st.number_input('The amount charged to the customer monthly', min_value=0, max_value=150, value=0)
        totalcharges = st.number_input('The total amount charged to the customer',min_value=0, max_value=10000, value=0)

        st.subheader("Services signed up for")
        mutliplelines = st.selectbox("Does the customer have multiple lines",('Yes','No','No phone service'))
        phoneservice = st.radio(
            "Phone Service 👉",
            options=["Yes", "No"],
        )
        internetservice = st.selectbox("Does the customer have internet service", ('No', 'DSL', 'Fiber optic'))

        if internetservice != 'No':
            onlinesecurity = st.selectbox("Does the customer have online security",  ('Yes', 'No'))
            onlinebackup = st.selectbox("Does the customer have online backup", ('Yes', 'No'))
            deviceprotection = st.selectbox("Does the customer have device protection", ('Yes', 'No'))
            techsupport = st.selectbox("Does the customer have technology support", ('Yes', 'No'))
            streamingtv = st.selectbox("Does the customer stream TV", ('Yes', 'No'))
            streamingmovies = st.selectbox("Does the customer stream movies", ('Yes', 'No'))
        else:
            onlinesecurity = onlinebackup = techsupport = streamingtv = streamingmovies = deviceprotection = 'No internet service'

        if seniorcitizen=='Yes':
            seniorcitizen=1
        else:
            seniorcitizen=0
        data = {
                'gender': gender,
                'SeniorCitizen': seniorcitizen,
                'Partner': partner,
                'Dependents': dependents,
                'tenure':tenure,
                'PhoneService': phoneservice,
                'MultipleLines': mutliplelines,
                'InternetService': internetservice,
                'OnlineSecurity': onlinesecurity,
                'OnlineBackup': onlinebackup,
                'DeviceProtection': deviceprotection,
                'TechSupport': techsupport,
                'StreamingTV': streamingtv,
                'StreamingMovies': streamingmovies,
                'Contract': contract,
                'PaperlessBilling': paperlessbilling,
                'PaymentMethod':PaymentMethod,
                'MonthlyCharges': monthlycharges,
                'TotalCharges': totalcharges
                }
        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)
        # Preprocess inputs
        # preprocess_df = preprocess(features_df,"online")


        if st.button('Predict'):
            prediction = model.predict(features_df)
            st.markdown("vnfjgfg")
            st.markdown(prediction)
            if prediction == 1:
                st.warning('Yes, the customer will terminate the service.')
            else:
                st.success('No, the customer is happy with Telco Services.')


    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            #Get overview of data
            st.write(data.head())
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            #Preprocess inputs
            # preprocess_df = preprocess(data, "Batch")
            if st.button('Predict'):
                #Get batch prediction
                # prediction = model.predict(preprocess_df)
                # prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                # prediction_df = prediction_df.replace({1:'Yes, the customer will terminate the service.',
                #                                     0:'No, the customer is happy with Telco Services.'})

                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                # st.write(prediction_df)

if __name__ == '__main__':
        main()