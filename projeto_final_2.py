import streamlit as st
import pandas as pd
from pycaret.classification import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Carregar o modelo treinado usando PyCaret
def load_trained_model(file_path='modelo_lightgbm'):
    model = load_model(file_path)
    return model

# Função para carregar o CSV
def load_csv():
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    else:
        return None

# Função para pré-processamento
def preprocess_data(data):
    # Verifique se a coluna 'data_ref' está presente
    if 'data_ref' not in data.columns:
        raise ValueError("A coluna 'data_ref' está faltando nos dados de entrada.")
    
    # Certifique-se de que 'data_ref' está no formato correto
    data['data_ref'] = pd.to_datetime(data['data_ref'], errors='coerce')
    if data['data_ref'].isnull().any():
        raise ValueError("Existem valores nulos ou formatos inválidos em 'data_ref'.")

    # Definição das características numéricas e categóricas
    numeric_features = ['idade', 'tempo_emprego', 'qt_pessoas_residencia', 'renda']
    categorical_features = ['sexo', 'posse_de_veiculo', 'posse_de_imovel', 'tipo_renda', 
                            'educacao', 'estado_civil', 'tipo_residencia']

    # Transformadores para características numéricas e categóricas
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Usando mediana para imputar valores nulos
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Preprocessador para aplicar transformações
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Aplicar o pré-processamento
    data_preprocessed = preprocessor.fit_transform(data)
    return data_preprocessed

# Aplicar o modelo para escorar os dados
def score_data(model, data):
    predictions = model.predict(data)
    return predictions

# Interface do Streamlit
def main():
    # Configurar a página com o ícone e layout antes de outras chamadas
    st.set_page_config(
        page_title="Projeto Final EBAC",
        page_icon="https://github.com/digslima/ebac-image/blob/main/ebac_logo-data_science.png?raw=true",
        layout="wide"
    )
    
    st.title("Projeto Final EBAC")
    
    # Carregar o modelo
    model = load_trained_model()
    
    # Carregar o CSV
    data = load_csv()
    
    if data is not None:
        st.write("Dados carregados com sucesso!")
        
        # Pré-processar os dados
        try:
            data_preprocessed = preprocess_data(data)
        except ValueError as e:
            st.error(f"Erro no pré-processamento: {e}")
            return
        
        # Escorar os dados
        predictions = score_data(model, data_preprocessed)
        
        # Mostrar as previsões
        st.write("Previsões:")
        st.write(predictions)

if __name__ == "__main__":
    main()