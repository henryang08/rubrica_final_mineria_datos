import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(
    page_title="Clasificacion de Iris",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Dashboard para clasificar especies iris")
st.markdown("""
Este proyecto utiliza un modelo de **Random Forest** para clasificar flores Iris 
basado en las dimensiones de sus pétalos y sépalos.
""")

@st.cache_data 
def load_data():
    try:
        df = pd.read_csv("Iris.csv")
        
        if 'Id' in df.columns:
            df = df.drop(columns=['Id'])
            
        df.columns = df.columns.str.replace('Cm', ' (cm)')
        df.columns = df.columns.str.replace('SepalLength', 'sepal length')
        df.columns = df.columns.str.replace('SepalWidth', 'sepal width')
        df.columns = df.columns.str.replace('PetalLength', 'petal length')
        df.columns = df.columns.str.replace('PetalWidth', 'petal width')
        df.columns = df.columns.str.replace('Species', 'species')
        
        target_names = df['species'].unique()
        
        return df, target_names

    except FileNotFoundError:
        st.error("No se encontro el archivo 'Iris.csv'")
        return pd.DataFrame(), []

df, target_names = load_data()

if not df.empty:

    @st.cache_resource 
    def train_model(df):
        X = df.iloc[:, :-1] # Todas las columnas menos la última (species)
        y = df['species']   # La columna objetivo
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        return model, metrics

    model, metrics = train_model(df)

    st.sidebar.header("Ingresa las Dimensiones")
    st.sidebar.markdown("Modifica los valores para predecir la especie:")

    def user_input_features():
        sepal_length = st.sidebar.slider('Sepal length (cm)', 4.0, 8.0, 5.0)
        sepal_width = st.sidebar.slider('Sepal width (cm)', 2.0, 4.5, 3.4)
        petal_length = st.sidebar.slider('Petal length (cm)', 1.0, 7.0, 1.4)
        petal_width = st.sidebar.slider('Petal width (cm)', 0.1, 2.5, 0.2)
        
        data = {
            'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width
        }
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Metricas del Modelo")
        st.markdown("Rendimiento en el set de prueba:")
        
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
        c2.metric("F1 Score", f"{metrics['f1']:.2f}")
        
        c3, c4 = st.columns(2)
        c3.metric("Precision", f"{metrics['precision']:.2f}")
        c4.metric("Recall", f"{metrics['recall']:.2f}")
        
        st.info("Modelo utilizado: **Random Forest Classifier**")

    # Prediccion
    with col2:
        st.subheader("Prediccion en Tiempo Real")
        
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        st.write("Datos ingresados:")
        st.write(input_df)
        
        st.markdown(f"### La especie predicha es: **{prediction[0].upper()}**")
        
        # Barras de probabilidad
        proba_df = pd.DataFrame(prediction_proba, columns=target_names)
        st.bar_chart(proba_df.T)

    st.divider()

    # Grafico 3D
    st.subheader("Visualizacion 3D: Donde cae tu muestra")

    # Combinacion del dataset original con el del usuario
    df_copy = df.copy()
    df_copy['species'] = df_copy['species'].astype(str)
    input_df['species'] = 'USUARIO (Tu Input)' 

    df_viz = pd.concat([df_copy, input_df], ignore_index=True)

    # Selector de ejes para el grafico 3D
    c_x, c_y, c_z = st.columns(3)
    cols = df.columns[:-1] 
    x_axis = c_x.selectbox('Eje X', cols, index=2)
    y_axis = c_y.selectbox('Eje Y', cols, index=3)
    z_axis = c_z.selectbox('Eje Z', cols, index=0)

    fig_3d = px.scatter_3d(
        df_viz, 
        x=x_axis, 
        y=y_axis, 
        z=z_axis,
        color='species',
        symbol='species',
        opacity=0.8,
        title="Comparacion Espacial",
        color_discrete_map={'USUARIO (Tu Input)': 'red'} 
    )
    fig_3d.update_traces(marker=dict(size=5))
    st.plotly_chart(fig_3d, use_container_width=True)

    st.divider()

    # EDA
    with st.expander("Ver Visualizaciones Exploratorias (EDA)"):
        st.markdown("### Pairplot del Dataset Original")
        
        fig_sns = sns.pairplot(df, hue="species", palette="husl")
        st.pyplot(fig_sns)
        
        st.markdown("### Matriz de Correlación")
        # Filtramos solo columnas numericas
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        fig_corr, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig_corr)

else:
    st.warning("Esperando archivo de datos...")