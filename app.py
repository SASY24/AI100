import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# โหลดข้อมูล (ในที่นี้เราจะสร้างข้อมูลตัวอย่าง)
data = pd.DataFrame({
    'อายุ': np.random.randint(18, 60, 1000),
    'ประสบการณ์ทำงาน': np.random.randint(0, 40, 1000),
    'คะแนนการประเมิน': np.random.uniform(0, 5, 1000),
    'จำนวนชั่วโมงการอบรม': np.random.randint(0, 100, 1000),
    'เพศ_หญิง': np.random.randint(0, 2, 1000),
    'การศึกษา_ปริญญาโท': np.random.randint(0, 2, 1000),
    'การศึกษา_ปริญญาเอก': np.random.randint(0, 2, 1000),
    'ลาออก': np.random.randint(0, 2, 1000)  # 0 = ไม่ลาออก, 1 = ลาออก
})

# แยกข้อมูลเป็น X และ y
X = data.drop('ลาออก', axis=1)
y = data['ลาออก']

# แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ใช้ SMOTE เพื่อแก้ปัญหา imbalanced data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# สร้าง scaler และโมเดล
scaler = StandardScaler()
rf_model = RandomForestClassifier(random_state=42)

# กำหนด hyperparameters สำหรับ GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# ใช้ GridSearchCV เพื่อหา hyperparameters ที่ดีที่สุด
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(scaler.fit_transform(X_train_resampled), y_train_resampled)

# เลือกโมเดลที่ดีที่สุด
best_rf_model = grid_search.best_estimator_

# ฝึกโมเดลด้วย hyperparameters ที่ดีที่สุด
X_train_scaled = scaler.transform(X_train_resampled)
best_rf_model.fit(X_train_scaled, y_train_resampled)

# ประเมินประสิทธิภาพของโมเดล
X_test_scaled = scaler.transform(X_test)
y_pred = best_rf_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# ฟังก์ชันทำนายความเสี่ยงการลาออก
def predict_turnover_risk(employee_data):
    employee_scaled = scaler.transform(employee_data)
    risk_prob = best_rf_model.predict_proba(employee_scaled)[0, 1]
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

# แสดงประสิทธิภาพของโมเดล
st.write("---")
st.write("ประสิทธิภาพของโมเดล:")
st.write(f"Accuracy: {accuracy:.2f}")
st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")
st.write(f"F1-score: {f1:.2f}")

# แสดง Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_rf_model.feature_importances_
}).sort_values('importance', ascending=False)

st.write("---")
st.write("ความสำคัญของแต่ละปัจจัย:")
st.bar_chart(feature_importance.set_index('feature'))
