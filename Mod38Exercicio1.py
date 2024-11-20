#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)

# # Tarefa I
# 
# Neste projeto, estamos construindo um credit scoring para cartão de crédito, em um desenho amostral com 15 safras, e utilizando 12 meses de performance.
# 
# Carregue a base de dados ```credit_scoring.ftr```.

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from scipy.stats import ks_2samp
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_feather('credit_scoring.ftr')
df.head(59)


# In[2]:


df.dtypes


# In[3]:


df.head(59)


# ## Amostragem
# 
# Separe os três últimos meses como safras de validação *out of time* (oot).
# 
# Variáveis:<br>
# Considere que a variável ```data_ref``` não é uma variável explicativa, é somente uma variável indicadora da safra, e não deve ser utilizada na modelagem. A variávei ```index``` é um identificador do cliente, e também não deve ser utilizada como covariável (variável explicativa). As restantes podem ser utilizadas para prever a inadimplência, incluindo a renda.
# 

# In[4]:


df['data_ref'] = pd.to_datetime(df['data_ref'])

df = df.sort_values(by='data_ref')

ultimo_mes = df['data_ref'].max()
tres_meses_antes = ultimo_mes - pd.DateOffset(months=3)

df_oot = df[df['data_ref'] > tres_meses_antes]

df_in_time = df[df['data_ref'] <= tres_meses_antes]


print(f"Tamanho do dataset in time: {df_in_time.shape}")
print(f"Tamanho do dataset OOT: {df_oot.shape}")


# ## Descritiva básica univariada
# 
# - Descreva a base quanto ao número de linhas, número de linhas para cada mês em ```data_ref```.
# - Faça uma descritiva básica univariada de cada variável. Considere as naturezas diferentes: qualitativas e quantitativas.

# In[5]:


total_linhas = df.shape[0]
print(f"Número total de linhas: {total_linhas}")

linhas_por_mes = df.groupby(df['data_ref'].dt.to_period('M')).size()
print("\nNúmero de linhas para cada mês:")
print(linhas_por_mes)

variaveis_qualitativas = ['sexo', 'posse_de_veiculo', 'posse_de_imovel', 'tipo_renda', 'educacao', 'estado_civil', 'tipo_residencia', 'mau']
variaveis_quantitativas = ['idade', 'tempo_emprego', 'qt_pessoas_residencia', 'renda']

print("\nDescritiva das variáveis qualitativas:")
for var in variaveis_qualitativas:
    print(f"\n{var}:")
    print(df[var].value_counts())
    print(df[var].value_counts(normalize=True) * 100)

print("\nDescritiva das variáveis quantitativas:")
descritiva_quantitativas = df[variaveis_quantitativas].describe()
print(descritiva_quantitativas)


# In[6]:


total_linhas = df_oot.shape[0]
print(f"Número total de linhas: {total_linhas}")

linhas_por_mes = df_oot.groupby(df_oot['data_ref'].dt.to_period('M')).size()
print("\nNúmero de linhas para cada mês:")
print(linhas_por_mes)

variaveis_qualitativas = ['sexo', 'posse_de_veiculo', 'posse_de_imovel', 'tipo_renda', 'educacao', 'estado_civil', 'tipo_residencia', 'mau']
variaveis_quantitativas = ['idade', 'tempo_emprego', 'qt_pessoas_residencia', 'renda']

print("\nDescritiva das variáveis qualitativas:")
for var in variaveis_qualitativas:
    print(f"\n{var}:")
    print(df[var].value_counts())
    print(df[var].value_counts(normalize=True) * 100)

print("\nDescritiva das variáveis quantitativas:")
descritiva_quantitativas = df[variaveis_quantitativas].describe()
print(descritiva_quantitativas)


# ## Descritiva bivariada
# 
# Faça uma análise descritiva bivariada de cada variável

# In[7]:


variaveis_qualitativas = ['sexo', 'posse_de_veiculo', 'posse_de_imovel', 'tipo_renda', 'educacao', 'estado_civil', 'tipo_residencia', 'mau']
variaveis_quantitativas = ['idade', 'tempo_emprego', 'qt_pessoas_residencia', 'renda']

# 1. Análise entre duas variáveis qualitativas
print("\nAnálise entre variáveis qualitativas:")
for var1 in variaveis_qualitativas:
    for var2 in variaveis_qualitativas:
        if var1 != var2:
            print(f"\nTabela cruzada entre {var1} e {var2}:")
            tabela_cruzada = pd.crosstab(df[var1], df[var2], normalize='index')
            print(tabela_cruzada)
            print()

# 2. Análise entre uma variável qualitativa e uma quantitativa
print("\nAnálise entre variáveis qualitativa e quantitativa:")
for var_qual in variaveis_qualitativas:
    for var_quant in variaveis_quantitativas:
        print(f"\nEstatísticas de {var_quant} por categorias de {var_qual}:")
        estatisticas = df.groupby(var_qual)[var_quant].describe()
        print(estatisticas)
        print()

# 3. Análise entre duas variáveis quantitativas
print("\nCorrelação entre variáveis quantitativas:")
correlacao = df[variaveis_quantitativas].corr()
print(correlacao)



sns.pairplot(df[variaveis_quantitativas])
plt.show()


# ## Desenvolvimento do modelo
# 
# Desenvolva um modelo de *credit scoring* através de uma regressão logística.
# 
# - Trate valores missings e outliers
# - Trate 'zeros estruturais'
# - Faça agrupamentos de categorias conforme vimos em aula
# - Proponha uma equação preditiva para 'mau'
# - Caso hajam categorias não significantes, justifique

# #### 1. Tratamento de Valores Missing

# In[8]:


missing_data = df.isnull().sum()
print("Valores missing por variável:\n", missing_data)

quantitativas = ['idade', 'tempo_emprego', 'qt_pessoas_residencia', 'renda']
imputer_mediana = SimpleImputer(strategy='median')
df[quantitativas] = imputer_mediana.fit_transform(df[quantitativas])

qualitativas = ['sexo', 'posse_de_veiculo', 'posse_de_imovel', 'tipo_renda', 'educacao', 'estado_civil', 'tipo_residencia']
imputer_moda = SimpleImputer(strategy='most_frequent')
df[qualitativas] = imputer_moda.fit_transform(df[qualitativas])


# #### 2. Tratamento de Outliers

# In[9]:


for var in quantitativas:
    limite_superior = df[var].quantile(0.95)
    limite_inferior = df[var].quantile(0.05)
    df[var] = np.where(df[var] > limite_superior, limite_superior, df[var])
    df[var] = np.where(df[var] < limite_inferior, limite_inferior, df[var])


# #### 3. Tratamento de Zeros Estruturais

# In[10]:


# Identificando e tratando zeros estruturais
# Exemplo: Se 'tempo_emprego' tiver zeros que significam falta de emprego formal, mantemos como está


# #### 4. Agrupamento de Categorias

# In[11]:


df['estado_civil'] = df['estado_civil'].replace({'Separado': 'Outros', 'Viúvo': 'Outros', 'Desquitado': 'Outros'})


# In[12]:


df.head(20)


# #### 5. Desenvolvimento do Modelo: Regressão Logística

# In[13]:


X = df.drop(columns=['mau', 'data_ref', 'index'])
y = df['mau']

X = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelo = LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced')
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
y_prob = modelo.predict_proba(X_test)[:, 1]

print("Relatório de Classificação:\n", classification_report(y_test, y_pred, zero_division=0))
print("AUC-ROC:", roc_auc_score(y_test, y_prob))


# ## Avaliação do modelo
# 
# Avalie o poder discriminante do modelo pelo menos avaliando acurácia, KS e Gini.
# 
# Avalie estas métricas nas bases de desenvolvimento e *out of time*.

# #### 1. Preparação dos Dados

# In[14]:


X = df.drop(columns=['mau', 'data_ref', 'index'])
y = df['mau']

X = pd.get_dummies(X, drop_first=True)

colunas = X.columns

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = pd.DataFrame(X, columns=colunas)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelo = LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced')
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
y_prob = modelo.predict_proba(X_test)[:, 1]

ultimo_mes = df['data_ref'].max()
tres_meses_antes = ultimo_mes - pd.DateOffset(months=3)
df_oot = df[df['data_ref'] > tres_meses_antes].copy()
df_in_time = df[df['data_ref'] <= tres_meses_antes].copy()

X_oot = df_oot.drop(columns=['mau', 'data_ref', 'index'])
y_oot = df_oot['mau']

X_oot = pd.get_dummies(X_oot, drop_first=True)

X_oot = X_oot.reindex(columns=colunas, fill_value=0)

X_oot = scaler.transform(X_oot)

y_pred_oot = modelo.predict(X_oot)
y_prob_oot = modelo.predict_proba(X_oot)[:, 1]

# Métricas na base de teste
acuracia_test = accuracy_score(y_test, y_pred)
gini_test = 2 * roc_auc_score(y_test, y_prob) - 1
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_prob)
ks_test = max(tpr_test - fpr_test)

print(f"Acurácia na base de teste: {acuracia_test:.4f}")
print(f"Gini na base de teste: {gini_test:.4f}")
print(f"KS na base de teste: {ks_test:.4f}")

# Métricas na base OOT
acuracia_oot = accuracy_score(y_oot, y_pred_oot)
gini_oot = 2 * roc_auc_score(y_oot, y_prob_oot) - 1
fpr_oot, tpr_oot, thresholds_oot = roc_curve(y_oot, y_prob_oot)
ks_oot = max(tpr_oot - fpr_oot)

print(f"Acurácia na base OOT: {acuracia_oot:.4f}")
print(f"Gini na base OOT: {gini_oot:.4f}")
print(f"KS na base OOT: {ks_oot:.4f}")


# #### 2. Avaliação na Base de Desenvolvimento

# In[15]:


y_pred_test = modelo.predict(X_test)
y_prob_test = modelo.predict_proba(X_test)[:, 1]

acuracia_test = accuracy_score(y_test, y_pred_test)
print(f"Acurácia na base de teste: {acuracia_test:.4f}")

# KS 
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_prob_test)
ks_test = max(tpr_test - fpr_test)
print(f"KS na base de teste: {ks_test:.4f}")

# Gini
gini_test = 2 * roc_auc_score(y_test, y_prob_test) - 1
print(f"Gini na base de teste: {gini_test:.4f}")


# #### 3. Avaliação na Base Out of Time (OOT)

# In[16]:


y_pred_oot = modelo.predict(X_oot)
y_prob_oot = modelo.predict_proba(X_oot)[:, 1]

acuracia_oot = accuracy_score(y_oot, y_pred_oot)
print(f"Acurácia na base OOT: {acuracia_oot:.4f}")

fpr_oot, tpr_oot, thresholds_oot = roc_curve(y_oot, y_prob_oot)
ks_oot = max(tpr_oot - fpr_oot)
print(f"KS na base OOT: {ks_oot:.4f}")

gini_oot = 2 * roc_auc_score(y_oot, y_prob_oot) - 1
print(f"Gini na base OOT: {gini_oot:.4f}")


# In[17]:


df.qtd_filhos = df.qtd_filhos.astype(float)
df.dtypes


# ### a - Criar um pipeline utilizando o sklearn pipeline para o preprocessamento 

# ### Pré processamento

# ### Substituição de nulos (nans)
# 
# Existe nulos na base? é dado numérico ou categórico? qual o valor de substituição? média? valor mais frequente? etc

# ### Remoção de outliers
# 
# Como identificar outlier? Substituir o outlier por algum valor? Remover a linha?

# ### Seleção de variáveis
# 
# Qual tipo de técnica? Boruta? Feature importance? 

# ### Redução de dimensionalidade (PCA)
# 
# Aplicar PCA para reduzir a dimensionalidade para 5

# ### Criação de dummies
# 
# Aplicar o get_dummies() ou onehotencoder() para transformar colunas catégoricas do dataframe em colunas de 0 e 1. 
# - sexo
# - posse_de_veiculo
# - posse_de_imovel
# - tipo_renda
# - educacao
# - estado_civil
# - tipo_residencia

# ### Pipeline 
# 
# Crie um pipeline contendo essas funções.
# 
# preprocessamento()
# - substituicao de nulos
# - remoção outliers
# - PCA
# - Criação de dummy de pelo menos 1 variável (posse_de_veiculo)

# In[21]:


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline  # Importar pipeline do imblearn
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

# Carregando os dados
df = pd.read_feather('credit_scoring.ftr')

# Tratamento de datas (convertendo para ano, mês, etc.)
if 'data_coluna' in df.columns:  # Substitua 'data_coluna' pela sua coluna de data
    df['ano'] = pd.to_datetime(df['data_coluna']).dt.year
    df['mes'] = pd.to_datetime(df['data_coluna']).dt.month
    df['dia'] = pd.to_datetime(df['data_coluna']).dt.day
    df.drop('data_coluna', axis=1, inplace=True)

# Variáveis numéricas e categóricas
numeric_features = ['idade', 'tempo_emprego', 'qt_pessoas_residencia', 'renda']
categorical_features = ['sexo', 'posse_de_veiculo', 'posse_de_imovel', 'tipo_renda', 
                        'educacao', 'estado_civil', 'tipo_residencia']

# Pipeline para variáveis numéricas e categóricas
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Tratamento de nulos e outliers
    ('scaler', StandardScaler())  # Normalização
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Transformações aplicadas às colunas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Pipeline completo com SMOTE aplicado após o pré-processamento
model_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),  # Aplicação do SMOTE após o preprocessamento
    ('pca', PCA(n_components=5)),  # Avaliação do PCA
    ('classifier', RandomForestClassifier())  # Modelo de classificação
])

# Separação dos dados em treino e teste
X = df.drop('mau', axis=1)
y = df['mau']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinamento do modelo
model_pipeline.fit(X_train, y_train)

# Avaliação
y_pred = model_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))


# ### b - Pycaret na base de dados 
# 
# Utilize o pycaret para pre processar os dados e rodar o modelo **lightgbm**. Faça todos os passos a passos da aula e gere os gráficos finais. E o pipeline de toda a transformação.
# 
# 

# In[22]:


import pandas as pd

df = pd.read_feather('credit_scoring.ftr')
df.head()


# In[25]:


from pycaret.classification import *

# Inicialização do ambiente com a variável target 'mau'
clf = setup(data=df, target='mau', 
            numeric_features=['tempo_emprego'],  
            categorical_features=['sexo', 'posse_de_veiculo', 'posse_de_imovel', 'tipo_renda', 'educacao', 'estado_civil', 'tipo_residencia'],  # Colunas categóricas corretas
            normalize=True, 
            remove_outliers=True,
            imputation_type='simple')  


# In[26]:


best_model = compare_models()


# In[27]:


lgbm_model = create_model('lightgbm')


# In[28]:


tuned_model = tune_model(lgbm_model)


# In[29]:


plot_model(tuned_model, plot='auc')  # Exemplo: Curva AUC
plot_model(tuned_model, plot='confusion_matrix')  # Exemplo: Matriz de Confusão


# In[30]:


save_model(tuned_model, 'modelo_lightgbm')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




