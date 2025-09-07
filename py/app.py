import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==============================
# Configuração do App
# ==============================
st.set_page_config(page_title="Sistema de Recomendação de Crédito",
                   page_icon="💳", layout="centered")

st.title("💳 Sistema de Recomendação de Crédito")
st.write("Preencha os dados do cliente para receber recomendações de produtos financeiros.")

# ==============================
# Caminhos relativos (repo + pasta models)
# ==============================
base_dir = Path(__file__).resolve().parents[1]
models_dir = base_dir / "models"

modelo_path = models_dir / "modelo_knn.joblib"
encoders_path = models_dir / "encoders.joblib"
dataset_path = models_dir / "dataset_treino.csv"

# ==============================
# Carregar modelo e encoders
# ==============================
@st.cache_resource(show_spinner="Carregando modelo...")
def load_modelos():
    knn = joblib.load(modelo_path)
    le_dict = joblib.load(encoders_path)
    return knn, le_dict

knn, le_dict = load_modelos()

# ==============================
# Carregar dataset (com fallback de encoding)
# ==============================
@st.cache_data(show_spinner="Carregando dataset...")
def load_dataset():
    encodings = ["utf-8", "latin1", "ISO-8859-1", "cp1252"]
    for enc in encodings:
        try:
            return pd.read_csv(dataset_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    st.error("❌ Não foi possível carregar o dataset. Verifique encoding.")
    st.stop()

df = load_dataset()

# ==============================
# Produtos
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
def select_from_encoder(col_name, label):
    classes = list(le_dict[col_name].classes_)
    return st.selectbox(label, classes)

col1, col2 = st.columns(2)

with col1:
    idade = st.number_input("Idade", min_value=18, max_value=100, step=1)
    sexo = select_from_encoder("sexo", "Sexo")
    cor = select_from_encoder("cor", "Cor")
    casado = select_from_encoder("casado", "Casado?")
    qt_filhos = st.number_input("Quantidade de filhos", min_value=0, step=1)
    cidade = select_from_encoder("cidade", "Cidade")

with col2:
    renda = st.number_input("Renda mensal (R$)", min_value=0.0, step=100.0, format="%.2f")
    qt_carros = st.number_input("Quantidade de carros", min_value=0, step=1)
    qt_cart_cred = st.number_input("Quantidade de cartões de crédito", min_value=0, step=1)
    casa_propria = select_from_encoder("casa_propria", "Possui casa própria?")
    credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, step=10)
    endivid = st.number_input("Endividamento (%)", min_value=0, max_value=100, step=1)
    trabalha = select_from_encoder("trabalha", "Trabalha atualmente?")

# ==============================
# Botão de recomendação
# ==============================
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
        st.error(f"❌ Valor não visto no treino: {e}")
        st.stop()

    # Encontrar vizinhos mais próximos
    dist, indices = knn.kneighbors(data, n_neighbors=5)

    # Calcular recomendações
    recs = []
    for prod in produtos:
        if prod in df.columns:
            score = float(df.iloc[indices.flatten()][prod].mean())
            recs.append((prod, score))

    recs.sort(key=lambda x: x[1], reverse=True)

    # Mostrar recomendações
    st.subheader("✅ Produtos Recomendados")
    for prod, score in recs:
        st.write(f"- {prod}: {score:.2f}")

    # Mostrar vizinhos semelhantes
    with st.expander("👥 Vizinhos mais próximos"):
        st.write(pd.DataFrame({
            "idx_vizinho": indices.flatten(),
            "distancia": dist.flatten()
        }))
