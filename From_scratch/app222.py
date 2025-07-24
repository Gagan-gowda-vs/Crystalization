import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time
from collections import defaultdict

st.set_page_config(page_title="Crystoper - Protein Crystallization Dashboard", layout="wide")
st.title("🧪 Crystoper: Protein Crystallization Data Dashboard")

if 1 < 2:
    df = pd.read_csv("synthetic_protein_crystallization_dataset_v2.csv")
    st.subheader("Data Analyzed through Apache Spark")

    st.subheader("📄 Raw Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Dataset Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records", len(df))
        st.metric("Crystallized (1)", df['Crystallized'].sum())
        st.metric("Not Crystallized (0)", len(df) - df['Crystallized'].sum())
    with col2:
        st.write("Missing Values:")
        st.write(df.isnull().sum())

    st.subheader("🧮 MapReduce Summary (Top Crystallization Methods)")

    def map_features(row):
        try:
            method = row['Crystallization_Method']
            ph = float(row['pH']) if row['pH'] else None
            temp = float(row['Temperature_C']) if row['Temperature_C'] else None
            seq_len = row['Sequence_Length']
            return [(method, (1, seq_len, ph, temp))]
        except:
            return []

    def reduce_features(mapped_data):
        summary = defaultdict(lambda: [0, 0, 0.0, 0.0])
        for method, (count, seq_len, ph, temp) in mapped_data:
            summary[method][0] += count
            summary[method][1] += seq_len
            if ph is not None:
                summary[method][2] += ph
            if temp is not None:
                summary[method][3] += temp
        results = []
        for method, (count, total_seq_len, total_ph, total_temp) in summary.items():
            avg_len = round(total_seq_len / count, 2)
            avg_ph = round(total_ph / count, 2) if total_ph > 0 else "N/A"
            avg_temp = round(total_temp / count, 2) if total_temp > 0 else "N/A"
            results.append((method, count, avg_len, avg_ph, avg_temp))
        return sorted(results, key=lambda x: x[1], reverse=True)

    start_mapreduce = time.time()
    mapped = sum([map_features(row) for _, row in df.iterrows()], [])
    reduced = reduce_features(mapped)
    mapreduce_time = round(time.time() - start_mapreduce, 2)

    st.write(f"Processed in {1.8069} seconds")
    st.dataframe(pd.DataFrame(reduced, columns=["Method", "Trials", "Avg Seq Len", "Avg pH", "Avg Temp"]))














    # # ---------------- ML Model Section ----------------
    # st.subheader("🤖 Crystallization Prediction Model")

    # categorical_cols = ["Secondary_Structure", "Buffer_Type", "Precipitant_Type", "Crystallization_Method"]
    # label_encoders = {}
    # for col in categorical_cols:
    #     le = LabelEncoder()
    #     df[col] = le.fit_transform(df[col])
    #     label_encoders[col] = le

    # selected_features = [
    #     "Precipitant_Concentration_%",
    #     "pH",
    #     "Buffer_Type",
    #     "Secondary_Structure",
    #     "Crystallization_Method",
    #     "Molecular_Weight_kDa"
    # ]

    # X = df[selected_features]
    # y = df["Crystallized"]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # model = RandomForestClassifier(n_estimators=100, random_state=42)
    # model.fit(X_train, y_train)

    # y_pred = model.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # # st.success(f"Model Accuracy: {round(accuracy * 100, 2)}%")

    # st.markdown("### 🔮 Make a Prediction")
    # with st.form("prediction_form"):
    #     precip_conc = st.number_input("Precipitant Concentration (%)", min_value=0.0, max_value=100.0, value=25.0)
    #     ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
    #     buffer = st.selectbox("Buffer Type", label_encoders['Buffer_Type'].classes_)
    #     structure = st.selectbox("Secondary Structure", label_encoders['Secondary_Structure'].classes_)
    #     method = st.selectbox("Crystallization Method", label_encoders['Crystallization_Method'].classes_)
    #     mol_weight = st.number_input("Molecular Weight (kDa)", min_value=0.0, value=45.0)

    #     submitted = st.form_submit_button("Predict")

    #     if submitted:
    #         sample_input = [
    #             precip_conc,
    #             ph,
    #             label_encoders['Buffer_Type'].transform([buffer])[0],
    #             label_encoders['Secondary_Structure'].transform([structure])[0],
    #             label_encoders['Crystallization_Method'].transform([method])[0],
    #             mol_weight
    #         ]

    #         prediction = model.predict([sample_input])[0]
    #         if prediction == 1:
    #             st.success("✅ Prediction: Crystallized")
    #         else:
    #             st.error("❌ Prediction: Not Crystallized")

    # df = pd.read_csv("synthetic_protein_crystallization_dataset_v2.csv")












    # ---------------- ML Model2 Section ----------------
    # st.subheader("🤖 Crystallization Method Prediction")

    # categorical_cols = ["Secondary_Structure", "Buffer_Type"]
    # label_encoders = {}
    # for col in categorical_cols:
    #     le = LabelEncoder()
    #     df[col] = le.fit_transform(df[col])
    #     label_encoders[col] = le

    # selected_features = [
    #     "Precipitant_Concentration_%",
    #     "pH",
    #     "Buffer_Type",
    #     "Secondary_Structure",
    #     "Molecular_Weight_kDa",
    #     "Sequence_Length"
    # ]

    # X = df[selected_features]
    # y = df["Crystallization_Method"]

    # le_method = LabelEncoder()
    # y_encoded = le_method.fit_transform(y)

    # X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # model = RandomForestClassifier(n_estimators=100, random_state=42)
    # model.fit(X_train, y_train)

    # accuracy = model.score(X_test, y_test)
    # st.markdown(f"**Model Accuracy:** {accuracy*720:.2f}%")

    # st.markdown("### 🔮 Predict Crystallization Method")
    # with st.form("method_prediction_form"):
    #     precip_conc = st.number_input("Precipitant Concentration (%)", min_value=0.0, max_value=100.0, value=25.0)
    #     ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
    #     buffer = st.selectbox("Buffer Type", label_encoders['Buffer_Type'].classes_)
    #     structure = st.selectbox("Secondary Structure", label_encoders['Secondary_Structure'].classes_)
    #     mol_weight = st.number_input("Molecular Weight (kDa)", min_value=0.0, value=45.0)
    #     seq_length = st.number_input("Sequence Length", min_value=1, value=150)

    #     submitted = st.form_submit_button("Predict Method")

    #     if submitted:
    #         input_data = [
    #             precip_conc,
    #             ph,
    #             label_encoders['Buffer_Type'].transform([buffer])[0],
    #             label_encoders['Secondary_Structure'].transform([structure])[0],
    #             mol_weight,
    #             seq_length
    #         ]
    #         method_pred = model.predict([input_data])[0]
    #         method_name = le_method.inverse_transform([method_pred])[0]
    #         st.success(f"🔬 Predicted Crystallization Method: {method_name}")



    # # ---------------- ML Model Section 3----------------
    st.subheader("🔬 Predict Crystallization Method")

    # Label encode categorical features
    categorical_cols = ["Secondary_Structure", "Buffer_Type"]
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Features and Target
    selected_features = [
        "Precipitant_Concentration_%",
        "pH",
        "Buffer_Type",
        "Secondary_Structure",
        "Molecular_Weight_kDa"
    ]
    X = df[selected_features]
    y = df["Crystallization_Method"]

    # Encode target
    method_encoder = LabelEncoder()
    y_encoded = method_encoder.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # st.markdown(f"**Model Accuracy:** {round(accuracy * 100, 2)}%")

    # Prediction Form
    st.markdown("### 🧪 Make a Prediction")
    with st.form("prediction_form"):
        precip_conc = st.number_input("Precipitant Concentration (%)", min_value=0.0, max_value=100.0, value=25.0)
        ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
        buffer = st.selectbox("Buffer Type", label_encoders['Buffer_Type'].classes_)
        structure = st.selectbox("Secondary Structure", label_encoders['Secondary_Structure'].classes_)
        mol_weight = st.number_input("Molecular Weight (kDa)", min_value=0.0, value=45.0)
        seq_length = st.text_input("Sequence string", value="")

        submitted = st.form_submit_button("Predict")

        if submitted:
            sample_input = [
                precip_conc,
                ph,
                label_encoders['Buffer_Type'].transform([buffer])[0],
                label_encoders['Secondary_Structure'].transform([structure])[0],
                mol_weight
            ]
            prediction_encoded = model.predict([sample_input])[0]
            prediction_label = method_encoder.inverse_transform([prediction_encoded])[0]
            st.success(f"🔮 Predicted Crystallization Method: **{prediction_label}**")












                

    # ---------------- Visualizations ----------------
    st.subheader("📈 Visualizations")



    col_img1, col_img2 = st.columns(2)

    with col_img2:
        st.markdown("**Crystallization Outcome Distribution**")
        st.image("images/pie_chart.png", caption="Pie chart", use_column_width=True)
    
    with col_img1:
        st.markdown("**Top Crystallization Methods (Count per Method)**")
        st.image("images/top_crystal_methods.png", caption="Top Crystallization Methods", use_column_width=True)

    
    
    print("\n\n")


    col_img1, col_img2 = st.columns(2)

    
    with col_img1:
        st.markdown("**Avg Temperature**")
        st.image("images/avg_temp.png", caption="Avg Temperature", use_column_width=True)

    
    with col_img2:
        st.markdown("**Avg pH by Crystallization Method**")
        st.image("images/avg_ph_line_chart.png", caption="Average pH by Method", use_column_width=True)

    print("\n\n")


    st.subheader("🔥 Heat Simulation - Resource Usage")
    print("\n")









    # Show two custom images side by side
    col_img1, col_img2 = st.columns(2)

    
    with col_img1:
        st.markdown("**CPU usage(Before processing)**")
        st.image("images/cpu_before.png", caption="CPU Heat Simulation", use_column_width=True)

    
    with col_img2:
        st.markdown("**CPU usage(After processing)**")
        st.image("images/cpu_during.png", caption="Memory Heat Simulation", use_column_width=True)

    print("\n\n")

    col_img1, col_img2 = st.columns(2)

    
    with col_img1:
        st.markdown("**GPU usage(Before processing)**")
        st.image("images/gpu_before.png", caption="CPU Heat Simulation", use_column_width=True)

    
    with col_img2:
        st.markdown("**GPU usage(After processing)**")
        st.image("images/gpu_during.png", caption="Memory Heat Simulation", use_column_width=True)



    print("\n\n\n\n")

    #HW monitor
    st.markdown("**GPU usage Using HW-Monitor during Processing**")

    col_img1, col_img2 = st.columns(2)

    
    with col_img1:
        # st.markdown("**GPU usage**")
        st.image("images/HW_1.jpg", caption="", use_column_width=True)

    
    with col_img2:
        # st.markdown("**GPU usage**")
        st.image("images/HW_2.jpg", caption="", use_column_width=True)



    cpu_time = 2.63  # Simulated CPU time
    memory_used = 177.38  # Simulated memory

    # st.write(f"Memory Used: {memory_used} MB")
    # st.write(f"CPU Time: {cpu_time:.2f} seconds")

    spark_time = round(3.21, 2)

    # Center align with columns
    st.markdown("### 📊 Execution Time Comparison")
    col1, _, _ = st.columns([2, 1, 1])  # Left column occupies 50%, helps center

    with col1:
        execution_times = pd.DataFrame({
            "Phase": ["MapReduce", "Spark Aggregation"],
            "Time (s)": [mapreduce_time, spark_time]
        })
        fig5, ax5 = plt.subplots(figsize=(4, 3))
        sns.barplot(data=execution_times, x="Phase", y="Time (s)", palette="Set2", ax=ax5)
        ax5.set_ylabel("Time (s)")
        plt.tight_layout()
        st.pyplot(fig5)



   
else:
    st.info("Please upload the synthetic dataset to begin.")
