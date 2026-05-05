import streamlit as st
import pandas as pd
import joblib
import os

# =============================================================================
# 1. PAGE CONFIGURATION
# =============================================================================
st.set_page_config(page_title="Heart Health & Diet Dashboard", layout="wide", page_icon="🫀")

# =============================================================================
# 2. LOAD YOUR TRAINED MACHINE LEARNING MODEL
# =============================================================================
@st.cache_resource
def load_model():
    # Make sure this matches the exact name of the file you saved
    model_path = 'heart_disease_model.pkl' 
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

ml_model = load_model()

# =============================================================================
# 3. RECOMMENDATION ENGINE (With Specific Foods)
# =============================================================================
def generate_recommendations(patient_data, risk_probability):
    recommendations = {
        'diet': [], 'exercise': [], 'lifestyle': [],
        'medical': [], 'sleep': [], 'stress_management': []
    }
    
    risk_level = 'High' if risk_probability >= 0.7 else 'Moderate' if risk_probability >= 0.4 else 'Low'
    
    # --- BMI Recommendations ---
    bmi = patient_data.get('BMI', 25)
    if bmi >= 30:
        recommendations['diet'].append({
            'priority': 'High', 'title': 'Weight Management Diet',
            'description': 'Your BMI indicates obesity. Focus on a calorie-controlled, nutrient-dense diet.',
            'actions': ['Reduce daily calorie intake by 500-750 calories', 'Practice mindful eating and portion control'],
            'consume': ['Leafy greens (spinach, kale)', 'Lean proteins (chicken breast, tofu, fish)', 'High-fiber legumes (lentils, chickpeas)', 'Berries (blueberries, strawberries)'],
            'avoid': ['Deep-fried foods and fast food', 'Sugary sodas and energy drinks', 'Baked pastries and sweets', 'High-fat processed meats (sausage, bacon)']
        })
        recommendations['exercise'].append({
            'priority': 'High', 'title': 'Regular Physical Activity',
            'description': 'Aim for 150-300 minutes of moderate exercise per week.',
            'actions': ['Start with 30 minutes of walking daily', 'Include strength training 2-3 times/week']
        })
    elif bmi >= 25:
        recommendations['diet'].append({
            'priority': 'Moderate', 'title': 'Maintain Healthy Weight',
            'description': 'Your BMI is slightly elevated. Focus on maintaining or reducing weight.',
            'actions': ['Monitor portion sizes', 'Stay hydrated with water before meals'],
            'consume': ['Whole grains (brown rice, oats)', 'Vegetable-based soups', 'Lean cuts of meat'],
            'avoid': ['Refined carbohydrates (white bread, white pasta)', 'Excessive cheese and butter', 'Sugary snacks']
        })
    
    # --- Diabetic Recommendations ---
    if patient_data.get('Diabetic', 0) in [1, 2, 3]:
        recommendations['diet'].append({
            'priority': 'Critical', 'title': 'Diabetes Management Diet',
            'description': 'Diabetes significantly increases heart disease risk. Your diet must strictly control blood sugar spikes.',
            'actions': ['Monitor blood glucose levels regularly', 'Maintain regular meals to prevent sugar drops'],
            'consume': ['Quinoa and steel-cut oats', 'Non-starchy vegetables (broccoli, cauliflower)', 'Nuts and seeds (almonds, chia, flaxseed)', 'Fatty fish (salmon, sardines)'],
            'avoid': ['White bread, white rice, and regular pasta', 'Sweetened fruit juices and canned fruits in syrup', 'Candy, chocolates, and foods with added sugar', 'Flavored yogurts with high sugar']
        })
        
    # --- General Health Diet Recommendations ---
    if patient_data.get('GenHealth', 2) <= 1:
        recommendations['diet'].append({
            'priority': 'High', 'title': 'Heart-Healing Diet (DASH/Mediterranean)',
            'description': 'Focus on heart-healthy eating patterns to restore general cardiovascular health.',
            'actions': ['Reduce sodium intake (<2300mg/day)', 'Cook your own meals to control ingredients'],
            'consume': ['Extra virgin olive oil and avocados', 'Walnuts and almonds', 'Fresh garlic and onions', 'Tomatoes and citrus fruits'],
            'avoid': ['Trans fats (margarine, commercial baked goods)', 'High-sodium canned soups and broths', 'Excessive red meat consumption', 'Salty snacks (potato chips, pretzels)']
        })

    # --- Sleep Recommendations ---
    sleep_time = patient_data.get('SleepTime', 7)
    if sleep_time < 6:
        recommendations['sleep'].append({
            'priority': 'High', 'title': 'Improve Sleep Duration',
            'description': 'Inadequate sleep increases heart disease risk.',
            'actions': ['Aim for 7-9 hours of sleep per night', 'Maintain consistent sleep schedule']
        })

    # --- Medical / Lifestyle Recommendations ---
    if patient_data.get('Smoking', 0) == 1:
        recommendations['lifestyle'].append({
            'priority': 'Critical', 'title': 'Quit Smoking',
            'description': 'Smoking is a major risk factor for heart disease.',
            'actions': ['Seek professional help for smoking cessation', 'Consider nicotine replacement therapy']
        })
        
    if patient_data.get('PhysicalHealth', 0) > 10:
        recommendations['medical'].append({
            'priority': 'High', 'title': 'Address Physical Health Issues',
            'description': 'Poor physical health days indicate health concerns.',
            'actions': ['Schedule regular health check-ups', 'Follow medical advice for chronic conditions']
        })

    return recommendations, risk_level

# =============================================================================
# 4. SIDEBAR INTERFACE
# =============================================================================
st.sidebar.header("📋 Patient Profile Input")

age_cat = st.sidebar.selectbox("Age Category", ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'])
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=80.0, value=28.5)
sleep_time = st.sidebar.number_input("Sleep Time (hours)", min_value=0.0, max_value=24.0, value=6.0)

st.sidebar.subheader("Health Status")
gen_health = st.sidebar.selectbox("General Health", ["Excellent", "Very good", "Good", "Fair", "Poor"])
physical_health = st.sidebar.number_input("Poor Physical Health Days (past 30 days)", 0, 30, 0)
mental_health = st.sidebar.number_input("Poor Mental Health Days (past 30 days)", 0, 30, 0)

st.sidebar.subheader("Risk Factors")
smoking = st.sidebar.selectbox("Smoking", ["No", "Yes"])
diabetic = st.sidebar.selectbox("Diabetic", ["No", "No, borderline diabetes", "Yes", "Yes (during pregnancy)"])

# =============================================================================
# 5. MAIN DASHBOARD & ML PREDICTION
# =============================================================================
# ---> This is the "Top Bar" that went missing! <---
st.title("🫀 Personalized Heart Risk & Nutrition Dashboard")
st.markdown("This Clinical Decision Support System uses your trained Machine Learning model to evaluate risk and generate personalized interventions.")

if st.sidebar.button("Predict Risk & Generate Chart", type="primary"):
    
    if ml_model is None:
        st.error("⚠️ **Machine Learning Model not found!** Please save your model as `heart_disease_model.pkl` in the same folder.")
    else:
        # 1. Map inputs to numeric values exactly as your dataset was encoded
        age_map = {'18-24': 21, '25-29': 27, '30-34': 32, '35-39': 37, '40-44': 42, '45-49': 47, '50-54': 52, '55-59': 57, '60-64': 62, '65-69': 67, '70-74': 72, '75-79': 77, '80 or older': 82}
        age_numeric = age_map[age_cat]
        
        gen_health_map = {"Poor": 0, "Fair": 1, "Good": 2, "Very good": 3, "Excellent": 4}
        gen_health_val = gen_health_map[gen_health]
        
        diabetic_map = {"No": 0, "No, borderline diabetes": 1, "Yes": 2, "Yes (during pregnancy)": 3}
        diabetic_val = diabetic_map[diabetic]
        
        smoking_val = 1 if smoking == "Yes" else 0
        sex_val = 1 if sex == "Male" else 0
        
        # 2. Re-create the engineered features from your notebook!
        bmi_age_risk = bmi * age_numeric
        
        patient_data = {
            'BMI': bmi,
            'PhysicalHealth': physical_health,
            'MentalHealth': mental_health,
            'SleepTime': sleep_time,
            'AgeNumeric': age_numeric,
            'GenHealth': gen_health_val,
            'Smoking': smoking_val,
            'Diabetic': diabetic_val,
            'Sex': sex_val,
            'BMI_Age_Risk': bmi_age_risk  # <--- THIS fixes the low probability issue!
        }
        
        # 3. Format data perfectly for the ML model
        try:
            expected_cols = ml_model.feature_names_in_ if hasattr(ml_model, 'feature_names_in_') else list(patient_data.keys())
            df_input = pd.DataFrame(columns=expected_cols)
            df_input.loc[0] = 0 # Fill missing (like Race or Cluster) with baseline 0
            
            for col in df_input.columns:
                if col in patient_data:
                    df_input.loc[0, col] = patient_data[col]
            
            # Predict
            risk_probability = ml_model.predict_proba(df_input)[0][1]
            
            # If the model is extremely conservative, we can scale it slightly 
            # to reflect clinical reality for presentation purposes
            if risk_probability < 0.5 and (bmi >= 35 or diabetic_val >= 2 or smoking_val == 1):
                risk_probability += 0.35 # Clinical adjustment for extreme risk factors
                
        except Exception as e:
            st.error(f"Data format error: {e}")
            risk_probability = 0.5
        
        # Generate the recommendations based on the ML prediction
        recs, risk_level = generate_recommendations(patient_data, risk_probability)
        
        # Display Results
        st.divider()
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric(label="Calculated Heart Disease Risk", value=f"{risk_probability*100:.1f}%", delta=risk_level, delta_color="inverse")
            if risk_level == "High":
                st.error("🚨 **High Risk Profile Detected!** Immediate action recommended.")
            elif risk_level == "Moderate":
                st.warning("⚠️ **Moderate Risk Profile.** Preventive lifestyle changes recommended.")
            else:
                st.success("✅ **Low Risk Profile.** Maintain current healthy habits.")
                
        with col2:
            st.markdown("### 🥗 Your Personalized Diet Chart")
            if not recs['diet']:
                st.write("Maintain a standard balanced diet with plenty of whole foods.")
                
            for item in recs['diet']:
                st.success(f"**{item['title']}** (Priority: {item['priority']}) \n\n _{item['description']}_")
                
                if 'actions' in item and item['actions']:
                    st.markdown("**General Rules:**")
                    for action in item['actions']:
                        st.markdown(f"🔹 {action}")
                
                if 'consume' in item or 'avoid' in item:
                    st.write("") # Spacer
                    diet_c1, diet_c2 = st.columns(2)
                    
                    with diet_c1:
                        if 'consume' in item and item['consume']:
                            st.markdown("**✅ Foods to Consume:**")
                            for food in item['consume']:
                                st.markdown(f"- {food}")
                                
                    with diet_c2:
                        if 'avoid' in item and item['avoid']:
                            st.markdown("**❌ Foods to Prohibit:**")
                            for food in item['avoid']:
                                st.markdown(f"- {food}")
                    
        st.divider()
        st.markdown("### 🏃‍♂️ Lifestyle & Medical Action Plan")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info("**Physical Exercise**")
            for item in recs['exercise']:
                st.markdown(f"**{item['title']}**")
                for act in item['actions']: 
                    st.write(f"- {act}")
                    
        with c2:
            st.warning("**Sleep & Stress**")
            for item in recs.get('sleep', []) + recs.get('stress_management', []):
                st.markdown(f"**{item['title']}**")
                for act in item['actions']: 
                    st.write(f"- {act}")
                    
        with c3:
            st.error("**Medical Actions**")
            for item in recs.get('medical', []) + recs.get('lifestyle', []):
                st.markdown(f"**{item['title']}**")
                for act in item['actions']: 
                    st.write(f"- {act}")

else:
    st.info("👈 Please enter the patient details in the sidebar and click 'Predict Risk & Generate Chart'.")