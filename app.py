import streamlit as st
from streamlit_plotly_events import plotly_events
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import datetime
from dateutil.parser import parse


# Helper to convert numbers like '1.5K', '2M'
def convert_to_number(val):
    if isinstance(val, str):
        val = val.strip()
        if val.endswith('K'):
            return float(val[:-1]) * 1_000
        elif val.endswith('M'):
            return float(val[:-1]) * 1_000_000
        elif val.endswith('B'):
            return float(val[:-1]) * 1_000_000_000
    try:
        return float(val)
    except:
        return np.nan

# Function to edit manual labels
def edit_labels(df, selected_indices):
    st.write(f"Selected indices: {selected_indices}")
    current_labels = df.loc[selected_indices, 'Manual_Label'].unique()
    st.write("Current Manual Labels:", current_labels)
    new_label = st.text_input("Enter new manual label for selected points:")
    if st.button("Update Labels"):
        df.loc[selected_indices, 'Manual_Label'] = new_label
        st.success(f"Updated labels for selected points to: {new_label}")
    return df

# Streamlit app
st.title("2D Data Labeling Tool")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # Force timezone removal on all datetime columns, regardless of format
    df = df.convert_dtypes()  # Ensures consistent internal dtype resolution
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.tz_localize(None)

    # Convert pandas 'string[python]' dtype to 'object' for compatibility
    for col in df.select_dtypes(include='string').columns:
        df[col] = df[col].astype(str)
    
    # --- Convert any 'object' (string-like) columns to datetime if possible ---

    def looks_like_datetime(series, sample_size=5):
        """Check if a string series likely represents datetime."""
        non_null = series.dropna().astype(str)
        sample = non_null.head(sample_size)
        success_count = 0
        for val in sample:
            try:
                # Use dateutil's parse (robust) — will work on many formats
                parse(val.strip())
                success_count += 1
            except:
                continue
        return success_count >= sample_size // 2  # at least half parse successfully

    # --- Apply datetime parsing only on likely date/time columns
    for col in df.select_dtypes(include='object').columns:
        if looks_like_datetime(df[col]):
            try:
                df[col] = pd.to_datetime(df[col],  errors='coerce')
            except:
                pass

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            try:
                df[col] = df[col].dt.tz_localize(None)
            except (AttributeError, TypeError):
                pass  # Column is already timezone-naive or not localizable


    
    # --- Convert number-looking strings (like '1.2K', '3M') to numeric ---
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].apply(convert_to_number)

    # st.write("Detected column types:")
    # st.write(df.dtypes)


    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())
    
    valid_cols = []
    for col in df.columns:
        try:
            # Check if the column is numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                valid_cols.append(col)
            # Check if the column is datetime
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                valid_cols.append(col)
        except TypeError:
            continue  # skip incompatible dtypes like string[python], etc.




    st.write("Detected valid columns for plotting:", valid_cols)

    if len(valid_cols) < 2:
        st.error("CSV must contain at least two numeric or datetime columns.")
    else:
        x_col = st.selectbox("Select X-axis column", valid_cols)
        y_col = st.selectbox("Select Y-axis column", valid_cols, index=1)



        # Drop NaNs
        df.dropna(subset=[x_col, y_col], inplace=True)

        # Clustering selection
        # algo = st.radio("Choose clustering algorithm", ("KMeans", "DBSCAN"))
        data = df[[x_col, y_col]]
        

        data_num = data.apply(pd.to_numeric, errors='coerce')
        data_num.dropna(inplace=True)

        # Standardize data
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data_num)

        # if algo == "KMeans":
        k = st.slider("K (for KMeans)", 2, 10, 3)
        model = KMeans(n_clusters=k)
        df["Cluster"] = model.fit_predict(scaled)
        df["Cluster"] = df["Cluster"].astype(str)
        # else:
        #     eps = st.slider("eps (for DBSCAN)", 0.1, 5.0, 0.5)
        #     min_samples = st.slider("min_samples (for DBSCAN)", 2, 10, 5)
        #     model = DBSCAN(eps=eps, min_samples=min_samples)
        #     df["Cluster"] = model.fit_predict(scaled)
        #     df["Cluster"] = df["Cluster"].astype(str)

        df['Manual_Label'] = df['Cluster'].astype(str)

        # Plot
        import plotly.colors as pc
        color_sequence=pc.qualitative.Alphabet
        fig = px.scatter(df, x=x_col, y=y_col, color='Cluster', color_discrete_sequence=color_sequence, title="🖱️ Select Points with Rectangular Tool")
        selected_points = plotly_events(fig, select_event=True, override_height=600, key="plot")

        if selected_points:
            selected_indices = [p['pointIndex'] for p in selected_points]
            df = edit_labels(df, selected_indices)

        # Download
        st.download_button(
            "Download Labeled CSV",
            data=df.to_csv(index=False),
            file_name="labeled_data.csv",
            mime="text/csv"
        )
