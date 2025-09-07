import streamlit as st
import pandas as pd
import joblib
import requests
import io

# Configuração da página
st.set_page_config(page_title="Sistema de Recomendação de Crédito", page_icon="💳", layout="centered")
st.title("💳 Sistema de Recomendação de Crédito")
st.write("Preencha os dados do cliente para receber recomendações de produtos financeiros.")

# URLs raw do GitHub
url_model = "https://raw.githubusercontent.com/RafaelGallo/Fiap_Sistema_recomendacao_credito/main/models/modelo_knn.joblib"
url_enc = "https://raw.githubusercontent.com/RafaelGallo/Fiap_Sistema_recomendacao_credito/main/models/encoders.joblib"
url_csv = "https://raw.githubusercontent.com/RafaelGallo/Fiap_Sistema_recomendacao_credito/main/input/data.csv"

# Função para baixar e carregar o modelo e encoders
@st.cache_resource
def load_model_and_encoders():
    resp1 = requests.get(url_model)
    knn = joblib.load(io.BytesIO(resp1.content))
    resp2 = requests.get(url_enc)
    le_dict = joblib.load(io.BytesIO(resp2.content))
    return knn, le_dict

knn, le_dict = load_model_and_encoders()

# Função para carregar o dataset com fallback de encoding
@st.cache_data
def load_dataset():
    resp = requests.get(url_csv)
    for enc in ["utf-8", "latin1", "ISO-8859-1", "cp1252"]:
        try:
            df = pd.read_csv(io.StringIO(resp.content.decode(enc)))
            return df
        except Exception:
            continue
    st.error("Não foi possível ler o dataset via URL. Verifique o encoding.")
    st.stop()

df = load_dataset()

# Lista de produtos — ajuste se os nomes mudarem
produtos = [
    "Conta Corrente Plus",
    "Cartão Platinum",
    "Seguro Residencial",
    "Crédito Pessoal Flex",
    "Investimento Renda Fixa"
]

# UI Inputs (baseado nas classes dos encoders)
def select_from(col, label):
    return st.selectbox(label, list(le_dict[col].classes_))

idade = st.number_input("Idade", min_value=18, max_value=100, step=1)
sexo = select_from("sexo", "Sexo")
cor = select_from("cor", "Cor")
casado = select_from("casado", "Casado?")
qt_filhos = st.number_input("Quantidade de filhos", min_value=0, step=1)
cidade = select_from("cidade", "Cidade")
renda = st.number_input("Renda mensal (R$)", min_value=0.0, step=100.0, format="%.2f")
qt_carros = st.number_input("Quantidade de carros", min_value=0, step=1)
qt_cart_cred = st.number_input("Quantidade de cartões de crédito", min_value=0, step=1)
casa_propria = select_from("casa_propria", "Possui casa própria?")
credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, step=10)
endivid = st.number_input("Endividamento (%)", min_value=0, max_value=100, step=1)
trabalha = select_from("trabalha", "Trabalha atualmente?")

# Ao clicar no botão, faz a recomendação
if st.button("🔮 Recomendar Produtos"):
    try:
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
            "trabalha": le_dict["trabalha"].transform([trabalha])[0],
        }])
    except ValueError as e:
        st.error(f"Erro na transformação de categoria: {e}")
        st.stop()

    dist, indices = knn.kneighbors(data, n_neighbors=5)
    vizinhos = indices.flatten()

    recs = []
    for prod in produtos:
        if prod in df.columns:
            score = df.iloc[vizinhos][prod].mean()
            recs.append((prod, score))
    recs.sort(key=lambda x: x[1], reverse=True)

    st.subheader("✅ Produtos Recomendados")
    for prod, score in recs:
        st.write(f"- {prod}: {score:.2f}")

    with st.expander("👥 Vizinhos mais próximos"):
        st.write(pd.DataFrame({"Índice": vizinhos, "Distância": dist.flatten()}))
