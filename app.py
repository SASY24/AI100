import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ตัวอย่างข้อมูลสำหรับการฝึก
data = pd.DataFrame({
    'อายุ': np.random.randint(18, 60, 100),
    'ประสบการณ์ทำงาน': np.random.randint(0, 40, 100),
    'คะแนนการประเมิน': np.random.uniform(0, 5, 100),
    'จำนวนชั่วโมงการอบรม': np.random.randint(0, 100, 100),
    'เพศ_หญิง': np.random.randint(0, 2, 100),
    'การศึกษา_ปริญญาโท': np.random.randint(0, 2, 100),
    'การศึกษา_ปริญญาเอก': np.random.randint(0, 2, 100),
    'ลาออก': np.random.randint(0, 2, 100)  # 0 = ไม่ลาออก, 1 = ลาออก
})

# แยกข้อมูลเป็น X และ y
X = data.drop('ลาออก', axis=1)
y = data['ลาออก']

# แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้าง scaler และโมเดล
scaler = StandardScaler()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# ฝึก scaler และโมเดล
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
rf_model.fit(X_train_scaled, y_train)

# ฟังก์ชันทำนายความเสี่ยงการลาออก
def predict_turnover_risk(employee_data):
    employee_scaled = scaler.transform(employee_data)
    risk_prob = rf_model.predict_proba(employee_scaled)[0, 1]
    risk_level = "ต่ำ" if risk_prob < 0.3 else "ปานกลาง" if risk_prob < 0.7 else "สูง"
    return risk_prob, risk_level

# ส่วนของ UI ด้วย Streamlit
st.title("การทำนายความเสี่ยงในการลาออกของพนักงาน")

# สร้าง input form ให้ผู้ใช้กรอกข้อมูลพนักงาน
age = st.number_input('อายุ', min_value=18, max_value=60, value=30)
experience = st.number_input('ประสบการณ์ทำงาน (ปี)', min_value=0, max_value=40, value=5)
performance_score = st.slider('คะแนนการประเมิน', min_value=0.0, max_value=5.0, value=3.0)
training_hours = st.number_input('จำนวนชั่วโมงการอบรม', min_value=0, max_value=100, value=20)
gender = st.selectbox('เพศ', options=['ชาย', 'หญิง'])
education = st.selectbox('การศึกษา', options=['ปริญญาตรี', 'ปริญญาโท', 'ปริญญาเอก'])

# เตรียมข้อมูลจากผู้ใช้
gender_female = 1 if gender == 'หญิง' else 0
education_master = 1 if education == 'ปริญญาโท' else 0
education_doctor = 1 if education == 'ปริญญาเอก' else 0

# รวบรวมข้อมูลทั้งหมดเป็น DataFrame
new_employee = pd.DataFrame({
    'อายุ': [age],
    'ประสบการณ์ทำงาน': [experience],
    'คะแนนการประเมิน': [performance_score],
    'จำนวนชั่วโมงการอบรม': [training_hours],
    'เพศ_หญิง': [gender_female],
    'การศึกษา_ปริญญาโท': [education_master],
    'การศึกษา_ปริญญาเอก': [education_doctor]
})

# แสดงข้อมูลที่ผู้ใช้กรอก
st.write("ข้อมูลพนักงานใหม่:")
st.write(new_employee)

# เมื่อผู้ใช้กดปุ่ม "ทำนาย"
if st.button("ทำนายความเสี่ยง"):
    risk_prob, risk_level = predict_turnover_risk(new_employee)
    st.write(f"ความเสี่ยงในการลาออก: {risk_prob:.2%}")
    st.write(f"ระดับความเสี่ยง: {risk_level}")

# สร้างรายงาน Classification Report เมื่อผู้ใช้กดปุ่ม "แสดง Classification Report"
if st.button("แสดง Classification Report"):
    # ทำการทำนายผลสำหรับชุดทดสอบ
    X_test_scaled = scaler.transform(X_test)  # ทำการปรับมาตราส่วนข้อมูลทดสอบ
    y_pred = rf_model.predict(X_test_scaled)   # ทำนายผลจากโมเดล

    # สร้าง Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # แสดงผลรายงาน
    st.write("Classification Report:")
    st.write(report)
