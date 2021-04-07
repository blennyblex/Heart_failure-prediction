import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
model = joblib.load("HFailure.pkl")

df = pd.read_csv("heart_failure_clinical_records_dataset.csv.xls")
df1 = df.drop("DEATH_EVENT", 1)

# try: 

st.sidebar.header("User Medical Records")
st.sidebar.subheader("Please enter your credentials here")


st.header("HEART_FAILURE PREDICTION")
st.text("This app predicts the likelihood of a person having an Heart Attack")

st.subheader("Insert the appropriate values using the input buttons by the left providing the following details: ")
st.text("""     
        1. age
        2. anaemia
        3. creatinine_phosphokinase (an enzyme found in your heart, brain, and skeletal muscles)
        4. diabetes
        5. ejection_fraction (percentage of blood leaving your heart each time it contracts)
        6. high_blood_pressure
        7. platelets
        8. serum_creatinine
        9. serum_sodium
        10.sex
        11.smoking
        12.time"""
)

def user_input():
    age = st.sidebar.number_input("What is your age?")
    anaemia = st.sidebar.selectbox("Do you have anaemia? ", (0, 1))
    creatinine_phosphokinase = st.sidebar.number_input("What is your creatinine_phosphokinase level?")
    diabetes = st.sidebar.selectbox("Do you have diabetes? ", (0, 1))
    ejection_fraction = st.sidebar.number_input("What is your ejection fraction?")
    high_blood_pressure = st.sidebar.selectbox("Do you have high blood pressure? ", (0, 1))
    platelets = st.sidebar.number_input("What is your Blood Platelets count?")
    serum_creatinine = st.sidebar.number_input("What is your body serum_creatinine level?")
    serum_sodium = st.sidebar.number_input("What is your body level of serum sodium?")
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


prediction = model.predict(df_)
st.subheader(body = "Prediction")

predict = pd.DataFrame({"DEATH_EVENT": prediction})
st.dataframe(predict)

if prediction == 0:
    st.write("This Patient is less likely to have an heart attack!!")
else:
    st.write("This Patient is suffering from Heart attack!!")

# except ValueError:
#     st.write("Please input correct values by the left")
