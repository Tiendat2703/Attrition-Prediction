import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


def scale_data(df):
    # Tách X và y
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    # Chuẩn hóa X
    scale = StandardScaler()
    X_scaled = scale.fit_transform(X)


    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    X_scaled_df['Attrition'] = y.reset_index(drop=True)
    X_scaled_df.to_csv('/Users/tiendat/Desktop/DW_final_dataset/scaled_with_label.csv', index=False)
    #return X_scaled, y

def smote_data(df):
    # Tách X và y
    X = df.drop("Attrition", axis=1)  
    y = df["Attrition"]              

    smote = SMOTE(random_state=42,sampling_strategy=0.7)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Chuyển lại thành DataFrame
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled["Attrition"] = y_resampled

    df_resampled.to_csv("/Users/tiendat/Desktop/DW_final_dataset/data_resampled.csv", index=False)

    print("Dữ liệu sau khi SMOTE:")
    print(df_resampled["Attrition"].value_counts())  
    
# Hàm PCA
def pca(X, y):
    pca = PCA()
    pca.fit(X)

    # Tỷ lệ phương sai tích lũy
    explained_variance_ratio = pca.explained_variance_ratio_.cumsum()

    # Vẽ biểu đồ tỷ lệ phương sai tích lũy
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o')
    plt.xlabel('Số thành phần chính')
    plt.ylabel('Tỷ lệ phương sai tích lũy')
    plt.title('Biểu đồ tỷ lệ phương sai tích lũy')
    plt.axhline(y=0.9, color='r', linestyle='--', label='Ngưỡng 90%')
    plt.legend()
    plt.grid()
    plt.show()

    # Chọn số thành phần tối ưu
    optimal_components = np.argmax(explained_variance_ratio >= 0.90) + 1
    print(f'Số chiều tối ưu: {optimal_components}')

    # Áp dụng PCA
    pca = PCA(n_components=optimal_components)
    pca_reduced = pca.fit_transform(X)

    # Tỷ lệ phương sai được giữ lại
    explained_variance = sum(pca.explained_variance_ratio_)
    print(f"Tỷ lệ phương sai được giữ lại: {explained_variance * 100:.2f}%")

    # Tạo DataFrame từ kết quả PCA và nối lại cột y
    pca_df = pd.DataFrame(pca_reduced, columns=[f'PC{i+1}' for i in range(optimal_components)])
    pca_df['Attrition'] = y.values
    pca_df.to_csv('/Users/tiendat/Desktop/DW_final_dataset/reduce_pca_smote.csv', index=False)
    print("PCA DataFrame saved as reduce_pca.csv")
    return pca_df


# Thực thi PCA và Random Forest
# Thực thi PCA và Random Forest
if __name__ == '__main__':
    #     # Đọc dữ liệu
    df = pd.read_csv('/Users/tiendat/Desktop/DW_final_dataset/scaled_with_label.csv')

    # Mã hóa các cột phân loại
    le = LabelEncoder()
    cat_col = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 
            'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']
    for col in cat_col:
        df[col] = le.fit_transform(df[col])
    smote_data(df)
    
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    pca(X, y)






