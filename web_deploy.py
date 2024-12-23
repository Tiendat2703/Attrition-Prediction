import streamlit as st 
import pandas as pd 
from function import models, preprocess

def main():
    data = pd.read_csv('/Users/tiendat/Downloads/WA_Fn-UseC_-HR-Employee-Attrition.csv')
    columns = list(data.columns)  
    randomforest,knn,knn_regres = models()

    if 'Attrition' in columns:
        columns.remove('Attrition')

    st.slider("Hello, welcome to web detect Attrition ability")
    st.title("Web Dự đoán khả năng nghỉ việc của nhân viên ")
    input_data = {}

    try:
        for col in columns:
            if col in ['Age','MonthlyIncome', 'DailyRate', 'DistanceFromHome', 'Education', 'HourlyRate',
                    'JobLevel', 'MonthlyRate', 'NumCompaniesWorked',
                    'PercentSalaryHike', 'PerformanceRating', 'StockOptionLevel',
                    'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
                    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
                    'YearsWithCurrManager']:
                input_data[col] = st.number_input(f"Nhập giá trị cho {col}", value=0)
            elif col in ['Over18', 'OverTime']:
            
                input_data[col] = st.selectbox(f"Chọn giá trị cho {col}", ['Yes', 'No'])
            elif col in ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']:
                
                options = data[col].unique() 
                input_data[col] = st.selectbox(f"Chọn giá trị cho {col}", options)
            else:
                input_data[col] = st.text_input(f"Nhập giá trị cho {col}")
                
        input_datas = pd.DataFrame([input_data])
        missing_cols = set(randomforest.feature_names_in_) - set(input_datas.columns)
        for col in missing_cols:
            input_datas[col] = 0
        input_datas = input_datas[randomforest.feature_names_in_] 
        input_datas_scaled = preprocess(input_datas)
        #input_datas_scaled1 = new_preprocess(input_datas)
        input_datas_scaled_knn = input_datas[
            ['Age', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
            'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'NumCompaniesWorked',
            'TotalWorkingYears', 'YearsAtCompany']
]

        if st.button("Dự đoán"):
            prediction_random = randomforest.predict(input_datas_scaled)
            prediction_knn = knn.predict(input_datas_scaled_knn )
            prediction_income = knn_regres.predict(input_datas_scaled_knn)
            st.write(f"Kết quả dự đoán khả năng nghỉ việc với randomforest: {'No' if prediction_random[0] == 0 else 'Yes'}")
            st.write(f"Kết quả dự đoán khả năng nghỉ việc với knn : {'No' if prediction_knn[0] == 0 else 'Yes'}")
            st.write(f"Kết quả dự đoán lương : {prediction_income[0]}")

    except Exception as e:
        st.error(f"Lỗi xảy ra: {e}")
if __name__ == '__main__':
    main()