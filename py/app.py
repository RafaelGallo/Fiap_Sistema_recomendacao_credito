import streamlit as st
import pandas as pd
import joblib
import requests
import io

# ==============================
# Configuração do App
# ==============================
st.set_page_config(page_title="Sistema de Recomendação de Crédito",
                   page_icon="💳", layout="centered")

st.title("💳 Sistema de Recomendação de Crédito")
st.write("Preencha os dados do cliente para receber recomendações de produtos financeiros.")

# ==============================
# URLs RAW do GitHub (ajuste se mudar o repo)
# ==============================
url_model = "https://raw.githubusercontent.com/RafaelGallo/Fiap_Sistema_recomendacao_credito/main/models/modelo_knn.joblib"
url_enc   = "https://raw.githubusercontent.com/RafaelGallo/Fiap_Sistema_recomendacao_credito/main/models/encoders.joblib"
url_csv   = "https://raw.githubusercontent.com/RafaelGallo/Fiap_Sistema_recomendacao_credito/main/input/data.csv"

# ==============================
# Carregar modelo e encoders
# ==============================
@st.cache_resource(show_spinner="Carregando modelo e encoders...")
def load_model_and_encoders():
    r1 = requests.get(url_model)
    r1.raise_for_status()
    knn = joblib.load(io.BytesIO(r1.content))

    r2 = requests.get(url_enc)
    r2.raise_for_status()
    le_dict = joblib.load(io.BytesIO(r2.content))
    return knn, le_dict

knn, le_dict = load_model_and_encoders()

# ==============================
# Carregar dataset (fallback de encoding)
# ==============================
@st.cache_data(show_spinner="Carregando dataset...")
def load_dataset():
    r = requests.get(url_csv)
    r.raise_for_status()
    for enc in ["utf-8", "latin1", "ISO-8859-1", "cp1252"]:
        try:
            return pd.read_csv(io.StringIO(r.content.decode(enc)))
        except Exception:
            continue
    st.error("❌ Não foi possível ler o dataset via URL.")
    st.stop()

df = load_dataset()

# ==============================
# Lista de produtos (ajuste se os nomes do CSV forem diferentes)
# ==============================
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
def select_from(col, label):
    return st.selectbox(label, list(le_dict[col].classes_))

col1, col2 = st.columns(2)
with col1:
    idade      = st.number_input("Idade", min_value=18, max_value=100, step=1)
    sexo       = select_from("sexo", "Sexo")
    cor        = select_from("cor", "Cor")
    casado     = select_from("casado", "Casado?")
    qt_filhos  = st.number_input("Quantidade de filhos", min_value=0, step=1)
    cidade     = select_from("cidade", "Cidade")
with col2:
    renda        = st.number_input("Renda mensal (R$)", min_value=0.0, step=100.0, format="%.2f")
    qt_carros    = st.number_input("Quantidade de carros", min_value=0, step=1)
    qt_cart_cred = st.number_input("Quantidade de cartões de crédito", min_value=0, step=1)
    casa_propria = select_from("casa_propria", "Possui casa própria?")
    credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, step=10)
    # Endividamento em porcentagem (UI)
    endivid_pct  = st.slider("Endividamento (%)", min_value=0, max_value=100, step=1, value=30)
    trabalha     = select_from("trabalha", "Trabalha atualmente?")

st.caption(f"Endividamento selecionado: **{endivid_pct}%**")

# ==============================
# Botão de recomendação
# ==============================
if st.button("🔮 Recomendar Produtos"):
    # Converte para fração (0–1) antes de mandar ao modelo
    endivid_fraction = endivid_pct / 100.0

    # Monta o registro com as MESMAS transformações do treino
    try:
        data = pd.DataFrame([{
            "idade":        idade,
            "sexo":         le_dict["sexo"].transform([sexo])[0],
            "cor":          le_dict["cor"].transform([cor])[0],
            "casado":       le_dict["casado"].transform([casado])[0],
            "qt_filhos":    qt_filhos,
            "cidade":       le_dict["cidade"].transform([cidade])[0],
            "renda":        renda,
            "qt_carros":    qt_carros,
            "qt_cart_cred": qt_cart_cred,
            "casa_propria": le_dict["casa_propria"].transform([casa_propria])[0],
            "credit_score": credit_score,
            "endivid":      endivid_fraction,  # <<< agora em fração
            "trabalha":     le_dict["trabalha"].transform([trabalha])[0],
        }])
    except ValueError as e:
        st.error(f"❌ Erro ao transformar categoria: {e}")
        st.stop()

    # Vizinhos mais próximos
    dist, indices = knn.kneighbors(data, n_neighbors=5)
    vizinhos = indices.flatten()

    # Recomendações por média dos vizinhos
    recs = []
    for prod in produtos:
        if prod in df.columns:
            score = float(df.iloc[vizinhos][prod].mean())
            recs.append((prod, score))
    recs.sort(key=lambda x: x[1], reverse=True)

    # Exibição
    st.subheader("✅ Produtos Recomendados")
    for prod, score in recs:
        st.write(f"**{prod}**")
        st.progress(min(score, 1.0))
        st.caption(f"Score: {score:.2f}")

    with st.expander("👥 Vizinhos mais próximos"):
        df_viz = pd.DataFrame({"Índice": vizinhos, "Distância": dist.flatten()})
        st.dataframe(df_viz.style.format({"Distância": "{:.4f}"}))
