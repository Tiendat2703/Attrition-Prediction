import pickle 
from sklearn.preprocessing import StandardScaler, LabelEncoder

def models():
    with open('random_pca.pkl', 'rb') as file:
        model = pickle.load(file) 
    with open('/Users/tiendat/Desktop/DW_final/knn_model.pkl', 'rb') as file:
        knn = pickle.load(file) 
    with open('/Users/tiendat/Desktop/DW_final/knn_regress.pkl', 'rb') as file:
        knn_regres = pickle.load(file) 
    return model , knn, knn_regres

def preprocess(input_datas):
        le = LabelEncoder()
        scale = StandardScaler()
        cat_col = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']
        for col in cat_col:
            if col in input_datas.columns:
                input_datas[col] = le.fit_transform(input_datas[col].fillna('Unknown'))  
        input_datas = input_datas.fillna(0)  
        input_datas_scaled = scale.fit_transform(input_datas)
        return input_datas_scaled
