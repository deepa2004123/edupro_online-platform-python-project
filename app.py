import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load("edupro_demand_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Title
st.title("📊 EduPro Course Demand Prediction System")

st.write("Predict course enrollment demand using Machine Learning")

# Sidebar inputs
st.sidebar.header("Enter Course Details")

course_price = st.sidebar.number_input("Course Price", 1000, 50000, 5000)
course_duration = st.sidebar.slider("Course Duration (hours)", 1, 200, 40)
course_rating = st.sidebar.slider("Course Rating", 1.0, 5.0, 4.0)
teacher_rating = st.sidebar.slider("Teacher Rating", 1.0, 5.0, 4.0)
experience = st.sidebar.slider("Instructor Experience (years)", 0, 30, 5)

course_category = st.sidebar.selectbox(
    "Course Category",
    [
        "Business",
        "Cybersecurity",
        "Data Science",
        "Design",
        "Digital Marketing",
        "Finance",
        "Machine Learning",
        "Marketing",
        "Programming",
        "Project Management",
        "Web Development"
    ]
)

course_level = st.sidebar.selectbox(
    "Course Level",
    ["Beginner", "Intermediate", "Advanced"]
)

# Feature Engineering
price_per_hour = course_price / course_duration
experience_rating_score = experience * teacher_rating
course_value = course_rating * course_price
demand_score = course_rating * 50

# Create input dictionary
input_dict = {
    "CoursePrice": course_price,
    "CourseDuration": course_duration,
    "CourseRating": course_rating,
    "YearsOfExperience": experience,
    "TeacherRating": teacher_rating,
    "Category_Avg_Enrollment": 50,
    "Price_Per_Hour": price_per_hour,
    "Experience_Rating_Score": experience_rating_score,
    "Course_Value": course_value,
    "Demand_Score": demand_score
}

# Add category encoding
input_dict[f"CourseCategory_{course_category}"] = 1

# Add level encoding
if course_level != "Advanced":
    input_dict[f"CourseLevel_{course_level}"] = 1

# Convert to dataframe
input_data = pd.DataFrame([input_dict])

# Match model columns exactly
input_data = input_data.reindex(columns=model_columns, fill_value=0)

# Prediction button
# Prediction button
if st.button("Predict Enrollment"):

    prediction = model.predict(input_data)[0]
    predicted_revenue = prediction * course_price

    st.subheader("Prediction Result")

    st.success(f"📈 Estimated Enrollment: {int(prediction)} students")

    # KPI Metrics
    st.subheader("📊 Course Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Course Price", f"₹ {course_price}")
    col2.metric("Course Duration", f"{course_duration} hrs")
    col3.metric("Predicted Enrollment", int(prediction))
    col4.metric("Revenue", f"₹ {int(predicted_revenue)}")

    # Demand level
    if prediction > 100:
        st.write("🔥 High Demand Course")

    elif prediction > 50:
        st.write("⚡ Medium Demand Course")

    else:
        st.write("📉 Low Demand Course")

    # Enrollment Chart
    st.subheader("📊 Enrollment Visualization")

    chart_data = pd.DataFrame({
        "Metric": ["Price", "Duration", "Predicted Enrollment"],
        "Value": [course_price, course_duration, prediction]
    })

    st.bar_chart(chart_data.set_index("Metric"))

    # Revenue Chart
    st.subheader("💰 Revenue Visualization")

    revenue_data = pd.DataFrame({
        "Metric": ["Predicted Enrollment", "Course Price", "Predicted Revenue"],
        "Value": [prediction, course_price, predicted_revenue]
    })

    st.line_chart(revenue_data.set_index("Metric"))
