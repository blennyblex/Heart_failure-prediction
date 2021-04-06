import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
model = joblib.load("HFailure.pkl")

df = pd.read_csv("heart_failure_clinical_records_dataset.csv.xls")
df1 = df.drop("DEATH_EVENT", 1)

# try: 

def user_input():
    st.header("Heart_Failure prediction")
    age = st.sidebar.number_input("What is your age")
    anaemia = st.sidebar.selectbox("Do you have anaemia? ", (0, 1))
    creatinine_phosphokinase = st.sidebar.number_input("What is your level of creatinine_phosphokinase")
    diabetes = st.sidebar.selectbox("Do you have diabetes? ", (0, 1))
    ejection_fraction = st.sidebar.number_input("What is your ejection fraction")
    high_blood_pressure = st.sidebar.selectbox("Do you have high blood pressure? ", (0, 1))
    platelets = st.sidebar.number_input("What is your Blood Platelets count")
    serum_creatinine = st.sidebar.number_input("What is your body level of serum_creatinine")
    serum_sodium = st.sidebar.number_input("What is your body level of serum sodium ")
    sex = st.sidebar.selectbox("What is your sex? Note: F = 1, M = 0", (0,1))
    smoking = st.sidebar.selectbox("Do you smoke?", (0, 1))
    time = st.sidebar.number_input("How many times have you gone for appointments?")



    data = pd.DataFrame({"age": age, "anaemia": anaemia, "creatinine_phosphokinase": creatinine_phosphokinase,
                        "diabetes": diabetes, "ejection_fraction": ejection_fraction, "high_blood_pressure":high_blood_pressure,
                        "platelets":platelets, "serum_creatinine": serum_creatinine, "serum_sodium":serum_sodium, 
                        "sex": sex, "smoking": smoking, "time": time}, index = [0])
    return data


def scaling():
    features = user_input()

    st.subheader(body = "User input")
    st.dataframe(features)


    df2 = pd.concat([features, df1], axis = 0)


    trans = pd.DataFrame(scaler.fit_transform(df2), columns=df2.columns)
    df_ = trans[:len(features)]
    
    return df_

df_ = scaling()

st.subheader(body = "Scaled User input")
st.dataframe(df_)

prediction = model.predict(df_)
st.subheader(body = "prediction")

predict = pd.DataFrame({"DEATH_EVENT": prediction})
st.dataframe(predict)

# except ValueError:
#     st.write("Please input correct values by the left")


