import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import os

# Label mapping
label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}

def plotTimeSeriesData(data):
    sample = data.loc[0, 'fft_0_b':'fft_749_b']
    plt.figure(figsize=(16, 10))
    plt.plot(range(len(sample)), sample)
    plt.title("Features fft_0_b through fft_749_b")
    plt.savefig("TSDplot.jpeg")
    plt.close()

mod = None

def load_model_from_file():
    global mod
    model_path_h5 = 'C:/Users/91939/Documents/vijay chandu/SEM Project/Project/FE/model-GRU.h5'
    model_path_tf = 'C:/Users/91939/Documents/vijay chandu/SEM Project/Project/FE/model-GRU'  # TensorFlow SavedModel directory

    # Check if it's an HDF5 model
    if os.path.isfile(model_path_h5):
        try:
            mod = load_model(model_path_h5)  # Load HDF5 model
            st.success("HDF5 model loaded successfully.")
        except Exception as e:
            st.error(f"Error loading HDF5 model: {e}")
    # Check if it's a TensorFlow SavedModel directory
    elif os.path.isdir(model_path_tf):
        try:
            mod = tf.keras.models.load_model(model_path_tf)  # Load TensorFlow SavedModel
            st.success("TensorFlow SavedModel loaded successfully.")
        except Exception as e:
            st.error(f"Error loading TensorFlow SavedModel: {e}")
    else:
        st.error("Model file not found or invalid format.")

def preprocess_inputs(df):
    df = df.copy()
    df['label'] = df['label'].replace(label_mapping)
    
    y = df['label'].copy()
    X = df.drop('label', axis=1).copy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=123)
    
    return X_train, X_test, y_train, y_test

def makePrediction(data, actual_data):
    predictions = mod.predict(data)
    rowNumberData = []
    predictionData = []
    actual_labels = actual_data['label'].replace(label_mapping).tolist()

    for i, res in enumerate(predictions):
        rowNumberData.append(i + range_values[0])
        predictionData.append(np.argmax(res))

    results = pd.DataFrame({
        "Row Number": rowNumberData,
        "Prediction Result": [list(label_mapping.keys())[pred] for pred in predictionData],
        "Actual Result": [list(label_mapping.keys())[label] for label in actual_labels]
    })

    model_acc = mod.evaluate(data, actual_data['label'].replace(label_mapping), verbose=0)[1]
    st.write("## Metrics")
    st.write("Test Accuracy: {:.3f}%".format(model_acc * 100))

    cm = confusion_matrix(actual_labels, predictionData)
    clr = classification_report(actual_labels, predictionData, target_names=list(label_mapping.keys()), output_dict=True)

    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='g', cbar=False, cmap='Blues')
    plt.xticks(np.arange(len(label_mapping)) + 0.5, label_mapping.keys())
    plt.yticks(np.arange(len(label_mapping)) + 0.5, label_mapping.keys())
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("cfm.jpeg")
    plt.close()

    st.write("Classification Report:")
    st.dataframe(clr)
    st.write("## Prediction Result:")
    st.write(results)

with st.spinner("Loading Model..."):
    load_model_from_file()

st.title("Emotion Detection using EEG Signals")

option = st.sidebar.radio("Select an option", ("Predict", "Download"))

if option == "Predict":
    st.header("Prediction Section")
    on = st.checkbox('Show Model Info')
    if on and mod:
        st.text(mod.summary(print_fn=lambda x: x))

    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        rows = df.shape[0]

        st.write("## Uploaded Data:")
        st.write(df)
        st.write("Dimensions: ", df.shape)
        plotTimeSeriesData(df)

        on = st.checkbox('Show data plot')
        if on:
            st.image("TSDplot.jpeg", caption='Time Series Data Plot')

        range_values = st.slider("Select a range", min_value=0, max_value=rows, value=(rows//4, (3*rows//4)))
        if range_values and range_values[0] != range_values[1]:
            if st.button(f"Make Prediction of subset {range_values}"):
                with st.spinner("Predicting...."):
                    makePrediction(df.iloc[range_values[0]:range_values[1]+1, :-1], df.iloc[range_values[0]:range_values[1]+1, :])
                st.write("# Plots")
                st.image("cfm.jpeg", caption='Confusion Matrix')
        elif range_values[0] == range_values[1]:
            st.warning('Please select at least two rows !!', icon="⚠️")

elif option == "Download":
    df = pd.read_csv('../emotions.csv')
    st.header("Download example Dataset:")
    with open('../emotions.csv.zip', 'rb') as f:
        st.download_button('Download Zip', f, file_name='emotions_Train.csv.zip')
