#### Projeto: Credit Scoring para Cartões de Crédito

**Descrição**

Este projeto consiste na construção de um modelo de credit scoring para cartões de crédito, utilizando dados de 15 safras amostrais e 12 meses de performance. O objetivo principal é prever a inadimplência de clientes a partir de variáveis socioeconômicas e comportamentais.

Estrutura do Projeto

Base de Dados: credit_scoring.ftr
 - Número total de registros: 750.000
 - Variáveis explicativas incluem dados como idade, renda, posse de bens, entre outras.

Divisão de dados:
- Dados "in time": 600.000 registros.
- Dados "out of time" (OOT): 150.000 registros para validação.

**Script Principal: "projeto_final_2.py"**
 - Desenvolvido com Streamlit para criar uma interface interativa.
 - Integra o modelo treinado usando PyCaret para realizar previsões.
 - Realiza pré-processamento dos dados, incluindo imputação de valores nulos e codificação de variáveis categóricas.
 - Permite carregamento de arquivos CSV e geração de previsões diretamente pela interface.

### Funcionalidades do Script Principal

1. Carregamento do Modelo Treinado:
 - Modelo treinado utilizando LightGBM com PyCaret.
 - Caminho padrão para o modelo: modelo_lightgbm.
2. Carregamento de Dados:
 - Upload de arquivos no formato CSV diretamente na interface do Streamlit.
3. Pré-processamento de Dados:
 - Imputação de valores nulos:
   - Mediana para variáveis numéricas.
   - Valor mais frequente para variáveis categóricas.
 - Padronização de variáveis numéricas.
 - Codificação one-hot para variáveis categóricas.
4. Predição:
 - Escoragem dos dados carregados usando o modelo treinado.
 - Exibição dos resultados na interface.
5. Interface Interativa:
 - Desenvolvida com Streamlit.
 - Layout amigável para uso por não técnicos.

### Pré-requisitos

Certifique-se de que os seguintes pacotes estão instalados:

 - streamlit
 - pandas
 - pycaret
- scikit-learn

Você pode instalar as dependências com:

 - pip install -r requirements.txt

### Como Executar

1 - Clone este repositório:
 - git clone [https://github.com/seu_usuario/nome_do_repositorio.git](https://github.com/digslima/ebac_projeto_final)
2 - Navegue até o diretório do projeto:
 - cd nome_do_repositorio
3 - Execute o script principal:
 - streamlit run projeto_final_2.py
4 - Acesse o aplicativo no navegador no endereço exibido pelo Streamlit (geralmente http://localhost:8501).

### Exemplo de Uso

1 - Abra o aplicativo.

2 - Faça o upload de um arquivo CSV contendo os dados dos clientes.

3 - Visualize os dados carregados na interface.

4 - Receba as previsões de inadimplência diretamente na tela.

### Contribuições

Sinta-se à vontade para contribuir com este projeto enviando pull requests ou abrindo issues no repositório.

