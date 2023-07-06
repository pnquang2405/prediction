from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import numpy as np
import uvicorn
import joblib

class Item(BaseModel):
    hight: int
    horizontal: int 
    width: int
    ratio: float
    name: str 

app = FastAPI()
model = None
name_encoder = ['hải châu', 'hồng ngọc mai', 'kim sa tùng', 'linh sam',
       'mai chiếu thuỷ', 'mai vàng', 'sam núi', 'sung']

async def load_model():
    global model
    try:
        model = joblib.load('./model.pkl')
        if model is not None:
            print("Đọc mô hình từ tệp joblib thành công!")
        else:
            print("Đọc mô hình từ tệp joblib thất bại!")
    except Exception as e:  
        print("Đọc mô hình từ tệp joblib thất bại:", e)

@app.on_event("startup")
async def startup_event():
    background_tasks = BackgroundTasks()
    background_tasks.add_task(load_model)
    print("Bắt đầu tải mô hình...")
    await background_tasks()

def replace_first_element(array, new_element):
    # Kiểm tra xem mảng có ít nhất một phần tử hay không
    if len(array) > 0:
        # Thay đổi giá trị của phần tử đầu tiên
        array[0] = new_element
    return array

def predict_linear_regression(model, input_vector):
    new_first_element = name_encoder.index(input_vector[0])

    new_input_array = replace_first_element(input_vector, new_first_element)
    print(new_input_array)
    new_input_array = np.array(new_input_array).reshape(1, -1)
    predicted_value = model.predict(new_input_array)
    
    return predicted_value

@app.get("/prediction/")
async def process_data(item: Item):
    global model
    if model is None:
        return {"message": "Đang tải mô hình, vui lòng đợi..."}
    input_vector = [item.name, item.hight, item.horizontal, item.width, item.ratio]
    predicted_value = predict_linear_regression(model, input_vector)[0]
    
    return {round(predicted_value,0)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
