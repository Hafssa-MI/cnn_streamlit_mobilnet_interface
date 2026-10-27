import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tensorflow import keras

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


st.set_page_config(page_title="Classification d'images naturelles", layout="wide")

IMG_SIZE = (128, 128)
MODEL_FILES = {
    "CNN base": "models/cnn_base.keras",
    "CNN amélioré": "models/cnn_ameliore.keras",
    "MobileNetV2 (TL)": "models/mobilenetv2_tl.keras",
}
COLORS = {
    "CNN base": "#6C8EBF",
    "CNN amélioré": "#E0A458",
    "MobileNetV2 (TL)": "#6DAA8A",
}
CLASSES_FR = {
    "airplane": "avion",
    "car": "voiture",
    "cat": "chat",
    "dog": "chien",
    "flower": "fleur",
    "fruit": "fruit",
    "motorbike": "moto",
    "person": "personne",
}

st.markdown(
    """
    <style>
    .stApp { background-color: #F5F7FA; color: #1F2937; }
    header[data-testid="stHeader"] { background-color: #F5F7FA; }
    h1, h2, h3, h4, p, label, li, span, div { color: #1F2937; }
    .block-container { padding-top: 2.5rem; max-width: 1200px; }
    button[data-baseweb="tab"] { color: #4B5563; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #2F6FA3; }
    div[data-baseweb="tab-highlight"] { background-color: #2F6FA3; }
    [data-testid="stFileUploaderDropzone"] { background-color: #FFFFFF; border: 1px dashed #B8C2CF; }
    [data-testid="stFileUploaderDropzone"] * { color: #374151; }
    [data-testid="stFileUploaderDropzone"] button { background-color: #FFFFFF; border: 1px solid #B8C2CF; }
    div[data-baseweb="select"] > div { background-color: #FFFFFF !important; border: 1px solid #D5DCE5 !important; }
    div[data-baseweb="select"] * { color: #1F2937 !important; -webkit-text-fill-color: #1F2937 !important; }
    div[data-baseweb="select"] svg { fill: #4B5563 !important; }
    div[data-baseweb="popover"], div[data-baseweb="popover"] > div, div[data-baseweb="menu"], ul[role="listbox"] { background-color: #FFFFFF !important; }
    li[role="option"] { background-color: #FFFFFF !important; }
    li[role="option"] * { color: #1F2937 !important; -webkit-text-fill-color: #1F2937 !important; }
    li[role="option"]:hover, li[role="option"][aria-selected="true"] { background-color: #EAF1F8 !important; }
    div[data-baseweb="popover"] *, [data-testid="stSelectboxVirtualDropdown"], [data-testid="stSelectboxVirtualDropdown"] * { background-color: #FFFFFF !important; color: #1F2937 !important; -webkit-text-fill-color: #1F2937 !important; }
    [data-testid="stSelectboxVirtualDropdown"] li:hover, [data-testid="stSelectboxVirtualDropdown"] li[aria-selected="true"], div[data-baseweb="popover"] li:hover { background-color: #EAF1F8 !important; }
    div[data-baseweb="popover"] { border: 1px solid #D5DCE5 !important; border-radius: 6px !important; }
    [data-testid="stMetric"] { background-color: #FFFFFF; border: 1px solid #E1E6EC; border-radius: 8px; padding: 14px 16px; }
    [data-testid="stMetricValue"] { color: #1F2937; }
    [data-testid="stMetricLabel"] * { color: #6B7280; }
    table { background-color: #FFFFFF; border: 1px solid #E1E6EC; }
    thead tr th { background-color: #EEF2F7; color: #1F2937; }
    tbody tr td, tbody tr th { background-color: #FFFFFF; color: #1F2937; }
    .card { background-color: #FFFFFF; border: 1px solid #E1E6EC; border-radius: 8px; padding: 16px 18px; margin-bottom: 10px; }
    .card .name { font-size: 0.85rem; color: #6B7280; letter-spacing: 0.02em; text-transform: uppercase; }
    .card .pred { font-size: 1.5rem; font-weight: 600; margin: 4px 0 2px 0; }
    .card .meta { font-size: 0.9rem; color: #4B5563; }
    .banner { background-color: #EAF1F8; border-left: 4px solid #2F6FA3; border-radius: 4px; padding: 12px 16px; margin: 10px 0 18px 0; }
    .subtitle { color: #6B7280; margin-top: -8px; margin-bottom: 18px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def fr(name):
    return CLASSES_FR.get(name, name)


def fmt_int(value):
    return f"{value:,}".replace(",", " ")


@st.cache_resource
def load_models():
    return {name: keras.models.load_model(path, compile=False) for name, path in MODEL_FILES.items()}


@st.cache_data
def load_info():
    with open("models/info.json", encoding="utf-8") as f:
        return json.load(f)


def open_image(file):
    image = Image.open(file)
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    return image.convert("RGB")


def preprocess(image):
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(image, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)


def run_models(models, x):
    results = {}
    for name, model in models.items():
        start = time.perf_counter()
        probs = model.predict(x, verbose=0)[0]
        results[name] = {"probs": probs, "ms": (time.perf_counter() - start) * 1000}
    return results


def style_axis(ax):
    ax.set_facecolor("#FFFFFF")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#C7CFDA")
    ax.tick_params(colors="#4B5563", labelsize=9)
    ax.grid(axis="y", color="#E8ECF1", linewidth=0.8)
    ax.set_axisbelow(True)


def new_figure(cols=1, width=12, height=3.8):
    fig, axes = plt.subplots(1, cols, figsize=(width, height))
    fig.patch.set_facecolor("#FFFFFF")
    axes_list = np.atleast_1d(axes)
    for ax in axes_list:
        style_axis(ax)
    return fig, axes_list


def show(fig):
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def page_prediction(models, classes):
    st.subheader("Tester une image")
    uploaded = st.file_uploader("Téléverser une image")
    st.caption("Classes : " + ", ".join(fr(c) for c in classes))
    if uploaded is None:
        return
    try:
        image = open_image(uploaded)
    except Exception:
        st.error("Ce fichier n'a pas pu être lu comme une image.")
        return
    results = run_models(models, preprocess(image))

    left, right = st.columns([1, 2.4], gap="large")
    with left:
        st.image(image.convert("RGB"), use_container_width=True)
    with right:
        cols = st.columns(len(results))
        for col, (name, res) in zip(cols, results.items()):
            probs = res["probs"]
            top = np.argsort(probs)[::-1][:3]
            with col:
                st.markdown(
                    f"""
                    <div class="card" style="border-top: 4px solid {COLORS[name]};">
                        <div class="name">{name}</div>
                        <div class="pred">{fr(classes[top[0]])}</div>
                        <div class="meta">Confiance {probs[top[0]] * 100:.1f}% &nbsp;|&nbsp; {res['ms']:.0f} ms</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                table = pd.DataFrame(
                    {"Classe": [fr(classes[i]) for i in top], "Probabilité": [f"{probs[i] * 100:.2f}%" for i in top]}
                )
                st.table(table.set_index("Classe"))

    predicted = {name: fr(classes[int(np.argmax(res["probs"]))]) for name, res in results.items()}
    if len(set(predicted.values())) == 1:
        message = f"Les trois modèles s'accordent sur la classe <b>{list(predicted.values())[0]}</b>."
    else:
        message = "Les modèles ne s'accordent pas : " + ", ".join(f"{n} prédit <b>{c}</b>" for n, c in predicted.items()) + "."
    st.markdown(f'<div class="banner">{message}</div>', unsafe_allow_html=True)

    fig, (ax,) = new_figure(1, 12, 3.8)
    width = 0.8 / len(results)
    x = np.arange(len(classes))
    for i, (name, res) in enumerate(results.items()):
        ax.bar(x + i * width - 0.4 + width / 2, res["probs"] * 100, width, label=name, color=COLORS[name])
    ax.set_xticks(x)
    ax.set_xticklabels([fr(c) for c in classes])
    ax.set_ylabel("Probabilité (%)", color="#4B5563")
    ax.set_title("Probabilités par classe et par modèle", loc="left", color="#1F2937", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    show(fig)


def page_comparison(info):
    st.subheader("Comparaison des modèles")
    rows = {}
    for name, m in info["models"].items():
        h = m["history"]
        rows[name] = {
            "Paramètres": fmt_int(m["params"]),
            "Paramètres entraînables": fmt_int(m["trainable_params"]),
            "Temps d'entraînement (s)": f"{m['train_time']:.1f}",
            "Accuracy entraînement": f"{h['accuracy'][-1]:.4f}",
            "Accuracy validation": f"{h['val_accuracy'][-1]:.4f}",
            "Accuracy test": f"{m['test_acc']:.4f}",
            "Loss test": f"{m['test_loss']:.4f}",
            "Écart entraînement - validation": f"{h['accuracy'][-1] - h['val_accuracy'][-1]:.4f}",
        }
    st.table(pd.DataFrame(rows).T)

    fig, axes = new_figure(2)
    for name, m in info["models"].items():
        axes[0].plot(range(1, len(m["history"]["val_accuracy"]) + 1), m["history"]["val_accuracy"], label=name, color=COLORS[name], linewidth=2)
        axes[1].plot(range(1, len(m["history"]["val_loss"]) + 1), m["history"]["val_loss"], label=name, color=COLORS[name], linewidth=2)
    axes[0].set_title("Accuracy de validation", loc="left", color="#1F2937", fontsize=11)
    axes[1].set_title("Loss de validation", loc="left", color="#1F2937", fontsize=11)
    for ax in axes:
        ax.set_xlabel("Époque", color="#4B5563")
        ax.legend(frameon=False, fontsize=9)
    show(fig)

    names = list(info["models"].keys())
    colors = [COLORS[n] for n in names]
    fig, axes = new_figure(3, 12, 3.4)
    metrics = [
        ("Accuracy test", [info["models"][n]["test_acc"] for n in names]),
        ("Temps d'entraînement (s)", [info["models"][n]["train_time"] for n in names]),
        ("Nombre de paramètres", [info["models"][n]["params"] for n in names]),
    ]
    for ax, (title, values) in zip(axes, metrics):
        ax.bar(range(len(names)), values, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_title(title, loc="left", color="#1F2937", fontsize=11)
    show(fig)


def page_evaluation(info, classes):
    st.subheader("Évaluation détaillée sur le jeu de test")
    name = st.selectbox("Modèle", list(info["models"].keys()))
    m = info["models"][name]
    report = m["report"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy test", f"{m['test_acc'] * 100:.2f}%")
    c2.metric("Loss test", f"{m['test_loss']:.4f}")
    c3.metric("Score F1 macro", f"{report['macro avg']['f1-score']:.3f}")
    c4.metric("Paramètres", fmt_int(m["params"]))

    h = m["history"]
    epochs = range(1, len(h["loss"]) + 1)
    fig, axes = new_figure(2)
    axes[0].plot(epochs, h["accuracy"], label="Entraînement", color="#6C8EBF", linewidth=2)
    axes[0].plot(epochs, h["val_accuracy"], label="Validation", color="#E0A458", linewidth=2)
    axes[0].set_title("Accuracy", loc="left", color="#1F2937", fontsize=11)
    axes[1].plot(epochs, h["loss"], label="Entraînement", color="#6C8EBF", linewidth=2)
    axes[1].plot(epochs, h["val_loss"], label="Validation", color="#E0A458", linewidth=2)
    axes[1].set_title("Loss", loc="left", color="#1F2937", fontsize=11)
    for ax in axes:
        ax.set_xlabel("Époque", color="#4B5563")
        ax.legend(frameon=False, fontsize=9)
    show(fig)

    labels = [fr(c) for c in classes]
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        cm = np.array(m["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        fig.patch.set_facecolor("#FFFFFF")
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9, color="#4B5563")
        ax.set_yticklabels(labels, fontsize=9, color="#4B5563")
        threshold = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=9,
                        color="#FFFFFF" if cm[i, j] > threshold else "#1F2937")
        ax.set_xlabel("Classe prédite", color="#4B5563")
        ax.set_ylabel("Classe réelle", color="#4B5563")
        ax.set_title("Matrice de confusion", loc="left", color="#1F2937", fontsize=11)
        for spine in ax.spines.values():
            spine.set_visible(False)
        show(fig)
    with right:
        table = pd.DataFrame(
            {
                "Précision": [report[c]["precision"] for c in classes],
                "Rappel": [report[c]["recall"] for c in classes],
                "Score F1": [report[c]["f1-score"] for c in classes],
                "Taux d'erreur": [1 - report[c]["recall"] for c in classes],
                "Images": [int(report[c]["support"]) for c in classes],
            },
            index=labels,
        )
        st.markdown("**Résultats par classe**")
        st.table(table.round(3))


st.title("Classification d'images naturelles")
st.markdown('<div class="subtitle">Comparaison de trois modèles convolutifs entraînés sur le jeu de données Natural Images</div>', unsafe_allow_html=True)

try:
    models = load_models()
    info = load_info()
except Exception as error:
    st.error(f"Impossible de charger les modèles ou le fichier info.json depuis le dossier models : {error}")
    st.stop()

classes = info["class_names"]

tab1, tab2, tab3 = st.tabs(["Prédiction", "Comparaison", "Évaluation"])
with tab1:
    page_prediction(models, classes)
with tab2:
    page_comparison(info)
with tab3:
    page_evaluation(info, classes)