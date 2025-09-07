import streamlit as st
import pandas as pd
import joblib
import os

# ==============================
# Configuração do App
# ==============================
st.set_page_config(page_title="Sistema de Recomendação de Crédito", page_icon="💳", layout="centered")

st.title("💳 Sistema de Recomendação de Crédito")
st.write("Preencha os dados do cliente para receber recomendações de produtos financeiros.")

# Caminho da pasta models
models_path = r"G:\Meu Drive\AI_data_lab\Cursos_ml_AI\Fiap\Recommendations Systems\Fiap_Sistema_recomendacao_credito\models"

# ==============================
# Carregar modelo e encoders
# ==============================
knn = joblib.load(os.path.join(models_path, "modelo_knn.joblib"))
le_dict = joblib.load(os.path.join(models_path, "encoders.joblib"))

# ==============================
# Carregar dataset com fallback de encoding
# ==============================
csv_path = os.path.join(models_path, r"G:\Meu Drive\AI_data_lab\Cursos_ml_AI\Fiap\Recommendations Systems\Fiap_Sistema_recomendacao_credito\input\data.csv")

encodings = ["utf-8", "latin1", "ISO-8859-1"]
df = None
for enc in encodings:
    try:
        df = pd.read_csv(csv_path, encoding=enc)
        print(f"✅ Dataset carregado com encoding {enc}")
        break
    except UnicodeDecodeError:
        continue

if df is None:
    st.error("❌ Não foi possível carregar o dataset. Verifique o encoding.")
    st.stop()

# Lista de produtos
produtos = [
    "Conta Corrente Plus",
    "Cartão Platinum",
    "Seguro Residencial",
    "Crédito Pessoal Flex",
    "Investimento Renda Fixa"
]

# ==============================
# Entradas do usuário
# ==============================
idade = st.number_input("Idade", min_value=18, max_value=100, step=1)
sexo = st.selectbox("Sexo", le_dict["sexo"].classes_)
cor = st.selectbox("Cor", le_dict["cor"].classes_)
casado = st.selectbox("Casado?", le_dict["casado"].classes_)
qt_filhos = st.number_input("Quantidade de filhos", min_value=0, step=1)
cidade = st.selectbox("Cidade", le_dict["cidade"].classes_)  # ou text_input
renda = st.number_input("Renda mensal (R$)", min_value=0.0, step=100.0, format="%.2f")
qt_carros = st.number_input("Quantidade de carros", min_value=0, step=1)
qt_cart_cred = st.number_input("Quantidade de cartões de crédito", min_value=0, step=1)
casa_propria = st.selectbox("Possui casa própria?", le_dict["casa_propria"].classes_)
credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, step=10)
endivid = st.number_input("Endividamento (%)", min_value=0, max_value=100, step=1)
trabalha = st.selectbox("Trabalha atualmente?", le_dict["trabalha"].classes_)

# ==============================
# Botão de recomendação
# ==============================
if st.button("🔮 Recomendar Produtos"):
    # Criar DataFrame com dados do cliente
    data = pd.DataFrame([{
        "idade": idade,
        "sexo": le_dict["sexo"].transform([sexo])[0],
        "cor": le_dict["cor"].transform([cor])[0],
        "casado": le_dict["casado"].transform([casado])[0],
        "qt_filhos": qt_filhos,
        "cidade": le_dict["cidade"].transform([cidade])[0],
        "renda": renda,
        "qt_carros": qt_carros,
        "qt_cart_cred": qt_cart_cred,
        "casa_propria": le_dict["casa_propria"].transform([casa_propria])[0],
        "credit_score": credit_score,
        "endivid": endivid,
        "trabalha": le_dict["trabalha"].transform([trabalha])[0]
    }])

    # Encontrar vizinhos mais próximos
    dist, indices = knn.kneighbors(data, n_neighbors=5)

    # Calcular recomendações
    recs = []
    for prod in produtos:
        score = df.iloc[indices.flatten()][prod].mean()
        recs.append((prod, score))

    recs = sorted(recs, key=lambda x: x[1], reverse=True)

    # Mostrar recomendações
    st.subheader("✅ Produtos Recomendados")
    for prod, score in recs:
        st.write(f"- {prod}: {score:.2f}")
