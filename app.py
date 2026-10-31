import streamlit as st

from src.predict import predict_customer


st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered",
)

st.title("Telco Customer Churn Predictor")
st.write("Enter the customer's service and billing details to estimate churn risk.")

with st.form("customer_details"):
    left, right = st.columns(2)

    with left:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox(
            "Senior citizen",
            [0, 1],
            format_func=lambda value: "Yes" if value == 1 else "No",
        )
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.number_input(
            "Tenure (months)", min_value=1, max_value=72, value=12, step=1
        )
        phone_service = st.selectbox("Phone service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple lines", ["No", "Yes", "No phone service"]
        )
        internet_service = st.selectbox(
            "Internet service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online security", ["No", "Yes", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online backup", ["No", "Yes", "No internet service"]
        )

    with right:
        device_protection = st.selectbox(
            "Device protection", ["No", "Yes", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech support", ["No", "Yes", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["No", "Yes", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming movies", ["No", "Yes", "No internet service"]
        )
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.number_input(
            "Monthly charges",
            min_value=0.0,
            value=70.35,
            step=0.50,
            format="%.2f",
        )
        total_charges = st.number_input(
            "Total charges",
            min_value=0.0,
            value=844.20,
            step=1.00,
            format="%.2f",
        )

    submitted = st.form_submit_button("Predict churn")

if submitted:
    customer = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    result = predict_customer(customer)

    st.subheader("Prediction")
    st.metric(
        "Estimated churn probability",
        f"{result['churn_probability']:.1%}",
    )

    if result["prediction"] == "Churn":
        st.warning("The model classifies this customer as likely to churn.")
    else:
        st.success("The model classifies this customer as likely to stay.")

    st.caption(
        f"Classification threshold: {result['threshold']:.2f}. "
        "This is a model estimate, not a certainty."
    )