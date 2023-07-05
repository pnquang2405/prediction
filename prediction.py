import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np


# df_train = pd.read_csv('/Users/laptoptt/Documents/Subject/data_set/data_price/train_data.csv')
# df_test = pd.read_csv('/Users/laptoptt/Documents/Subject/data_set/data_price/test_data.csv')

# name_mapping = {
#     'Tên bonsai': 'ten_bonsai',
#     'Chiều cao (cm)': 'chieu_cao_cm',
#     'Ngang (cm)': 'ngang_cm',
#     'Rộng thân (cm)': 'rong_than_cm',
#     'Tỉ lệ giống dáng (%)': 'ti_le_giong_dang',
#     'Giá': 'gia'
# }

# df_train = df_train.rename(columns=name_mapping)
# df_test = df_test.rename(columns=name_mapping)

# df_train.head()
# df_test.head()
# df_train.describe()

# df_train['ten_bonsai'].value_counts()
# df_test.describe()
# df_test['ten_bonsai'].value_counts()
# name_encoder = LabelEncoder()
# df_train['ten_bonsai_encode'] = name_encoder.fit_transform(df_train['ten_bonsai'])
# df_train.head()

# name_encoder.classes_
# name_encoder_arr = name_encoder.classes_
# mapping = {value: index for index, value in enumerate(name_encoder_arr)}
# df_test['ten_bonsai_encode'] = df_test['ten_bonsai'].map(mapping)
# df_test.head()

# df_train[['ten_bonsai_encode', 'chieu_cao_cm', 'ngang_cm', 'rong_than_cm','ti_le_giong_dang']].values

# X_train = df_train[['ten_bonsai_encode', 'chieu_cao_cm', 'ngang_cm', 'rong_than_cm','ti_le_giong_dang']].values
# X_test = df_test[['ten_bonsai_encode', 'chieu_cao_cm', 'ngang_cm', 'rong_than_cm','ti_le_giong_dang']].values
# y_train = df_train['gia'].values
# y_test = df_test['gia'].values

# # Tạo mô hình Linear Regression
# lr = LinearRegression()

# # Huấn luyện mô hình
# lr.fit(X_train, y_train)

# # Dự đoán trên tập kiểm tra
# y_pred = lr.predict(X_test)
# # Đánh giá mô hình bằng các chỉ số MAE, MSE, R2
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# # In ra các chỉ số đánh giá
# print("Mean Absolute Error (MAE):", mae)
# print("Mean Squared Error (MSE):", mse)
# print("R-squared (R2):", r2)

# with open('/Users/laptoptt/Documents/Subject/data_set/model/model.pkl', 'wb') as file:
#     pickle.dump(lr, file)

#Tải mô hình từ file
with open('/Users/laptoptt/Documents/Subject/data_set/model/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Khai báo map
name_encoder = ['hải châu', 'hồng ngọc mai', 'kim sa tùng', 'linh sam',
       'mai chiếu thuỷ', 'mai vàng', 'sam núi', 'sung']


def replace_first_element(array, new_element):
    # Kiểm tra xem mảng có ít nhất một phần tử hay không
    if len(array) > 0:
        # Thay đổi giá trị của phần tử đầu tiên
        array[0] = new_element
    return array


def predict_linear_regression(model, input_vector):
    # mapping = {value: index for index, value in enumerate(name_encoder)}
    new_first_element = name_encoder.index(input_vector[0])

    new_input_array = replace_first_element(input_vector, new_first_element)
    print(new_input_array)
    # Chuyển đổi đầu vào thành một mảng numpy có hình dạng (1, 5)
    new_input_array = np.array(new_input_array).reshape(1, -1)
    # Dự đoán giá trị đầu ra bằng mô hình Linear Regression
    predicted_value = model.predict(new_input_array)
    
    return predicted_value

input_vector = ["mai chiếu thuỷ", 38, 200, 15, 85.3]
predicted_value = predict_linear_regression(model, input_vector)[0]
print(f"Giá của cây có: \n+ Tên: {input_vector[0]} \n+ Chiều cao: {input_vector[1]} cm \n+ Chiều ngang: {input_vector[2]} cm\n+ Chiều rộng thân: {input_vector[3]} cm\n+ Tỉ lệ dáng giống: {input_vector[4]} % \nCó giá là: {round(predicted_value,0)} vnđ")