import streamlit as st
import pandas as pd
import numpy as np
import base64
import xgboost as xgb 
model = xgb. XGBRegressor()


model.load_model('xgb_model.json') 
from sklearn.preprocessing import LabelEncoder
st.markdown(
    """
    <div style="border: 2px solid #3498db; border-radius: 10px; padding: 10px; text-align: center;">
        <h1 style="font-size: 2em; color: #3498db;">التنبؤ بأسعار السيارات</h1>
    </div>
    """,
    unsafe_allow_html=True
)
image_path = r"C:\Users\Qandil\streamlit\giphy.gif"
with open(image_path, 'rb') as f:
    image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode()

st.markdown(
    f"""
    <div style="border: 2px solid #3498db; border-radius: 10px; padding: 10px; text-align: center;">
        <img src="data:image/gif;base64,{image_base64}" style="width: 100%; height: 100%;">
       
    </div>
    """,
    unsafe_allow_html=True
)
# قراءة البيانات من ملف نصي أو مصدر آخر
data = pd.read_csv('C:\\Users\\Qandil\\Downloads\\xgb_rit\\d.csv')  # استبدل بمسار ملف البيانات الخاص بك

# إنشاء قاموس لتخزين النماذج المرتبطة بكل شركة
models_dict = {}
for manufacturer in data['Manufacturer'].unique():

    models_dict[manufacturer] = list(data[data['Manufacturer'] == manufacturer]['Model'])

# اختيار الشركة
manufacturer = st.selectbox("اختر الشركة", data['Manufacturer'].unique())

# اختيار النموذج المرتبط بالشركة المحددة
selected_model = st.selectbox("اختر النموذج", models_dict[manufacturer])

# فلترة البيانات بناءً على الشركة والنموذج المحدد
filtered_data = data[(data['Manufacturer'] == manufacturer) & (data['Model'] == selected_model)]

# اختيار باقي المتغيرات
selected_category = st.selectbox("اختر الفئة", filtered_data['Category'].unique())
selected_leather_interior = st.selectbox("هل الداخلية جلدية؟", filtered_data['Leather interior'].unique())
selected_fuel_type = st.selectbox("اختر نوع الوقود", filtered_data['Fuel type'].unique())
selected_mileage = st.number_input("أدخل المسافة المقطوعة", min_value=0)
selected_gearbox_type = st.selectbox("اختر نوع صندوق التروس", filtered_data['Gear box type'].unique())
selected_drive_wheels = st.selectbox("اختر نوع المحرك", filtered_data['Drive wheels'].unique())
selected_wheel = st.selectbox("اختر نوع العجلة", filtered_data['Wheel'].unique())
selected_color = st.selectbox("اختر اللون", filtered_data['Color'].unique())
selected_levy = st.number_input("أدخل الرسوم", min_value=0)
selected_engine_volume = st.selectbox("اختر حجم المحرك", filtered_data['Engine volume'].unique())
selected_cylinders = st.slider("اختر عدد الأسطوانات", 4, 16, step=2)
#selected_cylinders = st.selectbox("اختر عدد الأسطوانات", filtered_data['Cylinders'].unique())
selected_airbags = st.number_input("أدخل عدد الوسائد الهوائية", min_value=0)
selected_age = st.number_input("أدخل العمر", min_value=0)


# عرض معلومات السيارة

# استخدام st.container لتكوين إطار
import streamlit as st

# عرض معلومات السيارة بتصميم أفضل
st.title("🚗 معلومات السيارة المدخلة:")

# تحسين التنسيق باستخدام الألوان والخطوط
st.markdown(
    """
    <div style="background-color: #3498db; padding: 20px; border-radius: 10px; text-align: center;">
        <h1 style="color: white; font-size: 2.5em; font-weight: bold;">معلومات السيارة</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# استخدام تخطيط الصفحة المنظم
col1, col2 = st.columns(2)

# عرض المعلومات في العمود الأول
with col1:
    st.subheader("المواصفات الأساسية:")
    st.write(f"**الشركة المصنعة:** {manufacturer}")
    st.write(f"**النموذج:** {selected_model}")
    st.write(f"**الفئة:** {selected_category}")
    st.write(f"**جلد الداخلية:** {selected_leather_interior}")
    st.write(f"**نوع الوقود:** {selected_fuel_type}")
    st.write(f"**المسافة المقطوعة:** {selected_mileage} كم")

# عرض المعلومات في العمود الثاني
with col2:
    st.subheader("التفاصيل التقنية:")
    st.write(f"**نوع صندوق التروس:** {selected_gearbox_type}")
    st.write(f"**نوع المحرك:** {selected_drive_wheels}")
    st.write(f"**نوع العجلة:** {selected_wheel}")
    st.write(f"**اللون:** {selected_color}")
    st.write(f"**الرسوم:** {selected_levy}")
    st.write(f"**حجم المحرك:** {selected_engine_volume} لتر")

# قسم إضافي
st.subheader("تكاليف ومالية:")
st.write(f"**عدد الأسطوانات:** {selected_cylinders}")
st.write(f"**عدد الوسائد الهوائية:** {selected_airbags}")
st.write(f"**العمر:** {selected_age} سنة")



original_encoding_dict = {}  # تعريف original_encoding_dict

# تحويل القيم باستخدام LabelEncoder لجميع الأعمدة باستثناء المتغيرات الرقمية
label_encoder_dict = {}
for column in data.select_dtypes(include='object').columns:
    label_encoder = LabelEncoder()
    data[column] = label_encoder.fit_transform(data[column])
    label_encoder_dict[column] = label_encoder
    original_encoding_dict[column] = label_encoder.classes_


data_new = pd.DataFrame({
    'Manufacturer': [label_encoder_dict['Manufacturer'].transform([manufacturer])[0]],
    'Model': [label_encoder_dict['Model'].transform([selected_model])[0]],
    'Category': [label_encoder_dict['Category'].transform([selected_category])[0]],
    'Leather interior': [label_encoder_dict['Leather interior'].transform([selected_leather_interior])[0]],
    'Fuel type': [label_encoder_dict['Fuel type'].transform([selected_fuel_type])[0]],
    'Mileage': [selected_mileage],
    'Gear box type': [label_encoder_dict['Gear box type'].transform([selected_gearbox_type])[0]],
    'Drive wheels': [label_encoder_dict['Drive wheels'].transform([selected_drive_wheels])[0]],
    'Wheel': [label_encoder_dict['Wheel'].transform([selected_wheel])[0]],
    'Color': [label_encoder_dict['Color'].transform([selected_color])[0]],
    'Levy': [selected_levy],
    'Engine volume': [selected_engine_volume],
    'Cylinders': [selected_cylinders],
    'Airbags': [selected_airbags],
    'Age': [selected_age]
}, index=[0])
       
if st.button("predict"):
    pr = model.predict(data_new)
    st.success(f'The expected price of your car. {pr[0]}')


