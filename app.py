"""Interface locale AyyPhoto : streamlit run app.py."""

import csv
import hashlib
from pathlib import Path
import time

import streamlit as st

from ayyphoto.config import EXPERIMENTS, PROJECT_ROOT, TrainConfig, read_json
from ayyphoto.download import DEFAULT_DESTINATION, inventory
from ayyphoto.imaging import decode_photo, png_bytes, preview_bytes
from ayyphoto.jobs import active_job, is_running, start_job, stop_job, tail_log
from ayyphoto.runtime import denoise, discover_local_models, load_model


st.set_page_config(page_title="AyyPhoto · Studio local", page_icon="📷", layout="wide")
st.markdown("""
<style>
    .stMainBlockContainer { max-width: 1260px; padding-top: 2rem; }
    [data-testid="stMetric"] { background: #eef2f6; border-radius: 12px; padding: 12px; }
    [data-testid="stSidebar"] { border-right: 1px solid #dce5eb; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(max_entries=2, show_spinner="Chargement du modèle…")
def cached_model(raw, label):
    return load_model(raw, label, threads=2)


def existing_models():
    return discover_local_models([PROJECT_ROOT / "artifacts", PROJECT_ROOT / "experiments"])


def local_image_folder(value):
    folder = Path(value).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError("Ce dossier n’existe pas sur cet ordinateur.")
    if not any(path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"} for path in folder.rglob("*")):
        raise ValueError("Aucune image JPEG/PNG trouvée dans ce dossier.")
    return str(folder)


def weights_panel():
    with st.sidebar:
        st.title("AyyPhoto")
        st.caption("Studio photo · sur cet ordinateur")
        st.subheader("Ton modèle")
        upload = st.file_uploader("Importer des poids ou un export", type=["pth", "pt", "ckpt", "zip"], key="model_file")
        paths = existing_models()
        selected = st.selectbox("Ou choisir un modèle local", [None, *paths],
                                format_func=lambda path: "Sélectionner…" if path is None else
                                (str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else path.name),
                                disabled=upload is not None)
        st.caption("Fichiers Kaggle acceptés : best_weights.pth, model_bundle.pt et archive d’export.")
        raw, label = None, None
        if upload is not None:
            raw, label = upload.getvalue(), upload.name
        elif selected is not None:
            if selected.stat().st_size > 64 * 1024 * 1024:
                st.error("Le modèle dépasse 64 Mio.")
            else:
                raw, label = selected.read_bytes(), selected.name
        loaded = None
        if raw is not None:
            try:
                loaded = cached_model(raw, label)
                st.success("Modèle chargé · CPU")
                with st.expander("Informations du modèle"):
                    st.write(loaded.label)
                    st.caption(f"Empreinte : {loaded.sha256[:16]}")
            except (ValueError, OSError, RuntimeError) as exc:
                st.error(str(exc))
        else:
            st.info("Importe tes poids entraînés pour corriger une photo.")
        st.divider()
        st.caption("La correction des photos est locale. Internet sert uniquement à préparer un dataset manquant ; aucun montage Drive n’est utilisé.")
        return loaded


def photo_panel(loaded):
    st.subheader("Une photo, moins de bruit")
    st.write("Choisis une photo, règle l’intensité et compare le résultat. Ton original reste intact.")
    photo = st.file_uploader("Photo à corriger", type=["jpg", "jpeg", "png"], key="photo_file")
    controls, note = st.columns([2, 1])
    with controls:
        strength = st.slider("Intensité du débruitage", 0, 100, 100, step=5, format="%d %%")
    with note:
        st.caption("Le rendu dépend de l’entraînement du modèle. Une intensité réduite peut préserver davantage de texture.")
    raw = photo.getvalue() if photo is not None else None
    request_key = hashlib.sha256(raw).hexdigest() + f":{loaded.sha256}:{strength}" if raw and loaded else None
    if request_key != st.session_state.get("photo_request"):
        st.session_state.pop("photo_result", None)
        st.session_state["photo_request"] = request_key
    if st.button("Corriger la photo", type="primary", disabled=not (raw and loaded), width="stretch"):
        try:
            original = decode_photo(raw)
            progress = st.progress(0.0, text="Traitement de la photo…")
            started = time.perf_counter()
            with st.spinner("Débruitage en cours…"):
                result = denoise(loaded, original, strength/100, tile_size=128,
                                 progress=lambda value: progress.progress(value, text="Traitement de la photo…"))
            progress.empty()
            st.session_state["photo_result"] = {
                "png": png_bytes(result), "before": preview_bytes(original), "after": preview_bytes(result),
                "seconds": time.perf_counter()-started, "width": result.shape[1], "height": result.shape[0],
                "filename": Path(photo.name).stem + "_ayyphoto.png",
            }
        except (ValueError, OSError, RuntimeError, FloatingPointError) as exc:
            st.error(str(exc))
    result = st.session_state.get("photo_result")
    if result:
        left, right = st.columns(2)
        left.image(result["before"], caption="Original", width="stretch")
        right.image(result["after"], caption="AyyPhoto", width="stretch")
        st.caption(f"{result['width']} × {result['height']} pixels · {result['seconds']:.1f} s · traitement CPU")
        st.download_button("Télécharger la photo corrigée", result["png"], result["filename"], "image/png", type="primary")
    elif photo is not None:
        try:
            original = decode_photo(raw)
            st.image(preview_bytes(original), caption="Photo prête à être traitée", width="stretch")
        except ValueError as exc:
            st.error(str(exc))
    else:
        st.info("Commence par importer ton modèle à gauche, puis une photo JPEG ou PNG.")


def data_panel():
    st.subheader("Préparer les images d’entraînement")
    st.write("DIV2K contient les images propres utilisées pour générer les paires bruitées et leurs références.")
    target = st.text_input("Dossier destination des images", str(DEFAULT_DESTINATION), key="data_target")
    st.caption("Répartition AyyPhoto : 720 images train, 80 validation et 100 réservées au test interne.")
    left, right = st.columns(2)
    if left.button("Vérifier la présence", width="stretch"):
        try:
            status = inventory(target)
            if status.complete:
                st.info("900 fichiers repérés. Le bouton de préparation vérifiera leur intégrité sans les retélécharger s’ils sont valides.")
            else:
                st.info(f"{len(status.paths)} fichiers repérés sur {len(status.expected)} attendus. La préparation récupérera les fichiers manquants ou invalides.")
        except (ValueError, OSError) as exc:
            st.error(str(exc))
    if right.button("Préparer les données DIV2K", type="primary", width="stretch"):
        try:
            if not target.strip():
                raise ValueError("Choisir un dossier de destination.")
            destination = Path(target).expanduser().resolve()
            inventory(destination)  # Refuser un dossier contenant d'autres images avant de lancer.
            job = start_job("data", {"data_root": str(destination)})
            st.session_state["selected_run"] = job["run_dir"]
            st.session_state["run_view"] = Path(job["run_dir"])
            st.success("Préparation lancée. Vérification et téléchargement sont suivis ci-dessous.")
        except (ValueError, OSError) as exc:
            st.error(str(exc))
    st.caption("Le premier téléchargement nécessite Internet. KaggleHub conserve un cache sur disque en plus du dossier final. Les prochains lancements réutilisent les images valides.")


def training_panel():
    st.subheader("Entraîner sur ton PC")
    st.write("Le calcul utilise le CPU et continue en arrière-plan pendant que l’interface reste ouverte.")
    source = st.radio("Données d’entraînement", ["DIV2K automatique", "Mon dataset local"], horizontal=True)
    automatic = source == "DIV2K automatique"
    if automatic:
        st.caption("Au lancement, les images sont vérifiées et les fichiers manquants sont téléchargés avant l’entraînement. Internet est nécessaire au premier téléchargement.")
    preset = st.radio("Type d’expérience", ["Essai rapide", "Entraînement complet"], horizontal=True)
    quick = preset == "Essai rapide"
    if quick:
        st.info("Essai limité à 40 images train et 8 images validation. Il vérifie le pipeline ; il ne valide pas la qualité finale.")
    else:
        st.info("Toutes les images seront utilisées. Sur le ThinkPad X270, commencer par mesurer une époque pour estimer la durée totale.")
    with st.form("train_form"):
        data_root = st.text_input("Dossier local des images", st.session_state.get("data_target", str(DEFAULT_DESTINATION)),
                                  help="Destination DIV2K en mode automatique ; sinon dossiers train/scene_… et val/scene_…")
        left, middle, right = st.columns(3)
        epochs = left.number_input("Époques", 1, 1000, 2 if quick else 30, key=f"epochs_{quick}")
        batch = middle.number_input("Images par lot", 1, 16, 1)
        patch = right.selectbox("Taille des recadrages", [32, 64, 128, 256], index=1 if quick else 2, key=f"patch_{quick}")
        with st.expander("Reprendre ou adapter un modèle"):
            initial = st.text_input("Poids initiaux (facultatif)", help="Chemin local des poids Kaggle pour commencer un nouvel entraînement.")
            resume = st.text_input("Dossier d’expérience locale à reprendre (facultatif)", help="Contient last.ckpt, best_weights.pth et config.json.")
            st.caption("Pour une reprise exacte, garder les réglages de l’expérience et augmenter le nombre total d’époques.")
        submit = st.form_submit_button("Lancer l’entraînement CPU", type="primary")
    if submit:
        try:
            if not data_root.strip():
                raise ValueError("Choisir un dossier pour les images.")
            root = str(Path(data_root).expanduser().resolve()) if automatic else local_image_folder(data_root)
            if automatic:
                inventory(root)
            cfg = TrainConfig(data_root=root, epochs=int(epochs), batch_size=int(batch), patch_size=int(patch),
                              eval_size=128 if quick else 256, patches_per_image=2 if quick else 8,
                              max_train_images=40 if quick else 0, max_val_images=8 if quick else 0,
                              initial_model=initial.strip(), resume_run=resume.strip(),
                              prepare_div2k=automatic).validate()
            job = start_job("train", cfg.to_dict())
            st.session_state["selected_run"] = job["run_dir"]
            st.session_state["run_view"] = Path(job["run_dir"])
            st.success("Vérification des données puis entraînement lancés. La progression apparaît ci-dessous." if automatic
                       else "Entraînement lancé. La progression et les sauvegardes apparaissent ci-dessous.")
        except (ValueError, OSError) as exc:
            st.error(str(exc))


def audit_panel():
    st.subheader("Vérifier tes données")
    st.write("Repère les fichiers illisibles, les dimensions insuffisantes et les doublons possibles, sans modifier les images.")
    with st.form("audit_form"):
        folder = st.text_input("Dossier des images à auditer", st.session_state.get("data_target", str(DEFAULT_DESTINATION)))
        submitted = st.form_submit_button("Lancer l’audit CPU")
    if submitted:
        try:
            job = start_job("audit", {"data_root": local_image_folder(folder)})
            st.session_state["selected_run"] = job["run_dir"]
            st.session_state["run_view"] = Path(job["run_dir"])
            st.success("Audit lancé. Le rapport sera disponible dans le suivi ci-dessous.")
        except (ValueError, OSError) as exc:
            st.error(str(exc))


@st.fragment(run_every=2)
def monitor():
    job = active_job()
    directories = sorted((path for path in EXPERIMENTS.glob("train_*") if path.is_dir()), reverse=True) if EXPERIMENTS.exists() else []
    if EXPERIMENTS.exists():
        directories += sorted((path for path in EXPERIMENTS.glob("audit_*") if path.is_dir()), reverse=True)
        directories += sorted((path for path in EXPERIMENTS.glob("data_*") if path.is_dir()), reverse=True)
    if not directories:
        return
    st.divider()
    st.subheader("Tes expériences locales")
    preferred = st.session_state.get("selected_run")
    index = next((i for i, value in enumerate(directories) if str(value) == preferred), 0)
    directory = st.selectbox("Expérience", directories, index=index, format_func=lambda value: value.name, key="run_view")
    status = read_json(directory / "status.json", {})
    running = bool(job and job.get("run_dir") == str(directory) and is_running(job))
    state = status.get("state", "unknown")
    message = status.get("message", state)
    if state == "completed":
        st.success(message)
    elif state == "error":
        st.error(message)
    elif state == "cancelled":
        st.warning(message)
    elif running:
        st.info(message)
    else:
        st.warning("Le calcul n’est plus actif. Consulte les journaux et les checkpoints disponibles.")
    if running:
        if status.get("total_steps"):
            st.progress(status.get("step", 0)/status["total_steps"])
        elif status.get("total"):
            st.progress(status.get("completed", 0)/status["total"])
        if st.button("Arrêter le calcul", key="stop_job"):
            try:
                if stop_job(job):
                    st.info("Arrêt demandé. Les checkpoints des époques complètes sont conservés.")
            except OSError as exc:
                st.error(str(exc))
    st.caption(f"Fichiers enregistrés : {directory}")
    history = directory / "history.csv"
    if history.is_file():
        try:
            with history.open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            if rows:
                left, middle, right = st.columns(3)
                left.metric("Époques terminées", len(rows))
                middle.metric("Meilleur PSNR validation", f"{max(float(row['val_psnr']) for row in rows):.2f} dB")
                right.metric("Dernier SSIM", f"{float(rows[-1]['val_ssim']):.3f}")
                st.line_chart({"PSNR validation (dB)": [float(row["val_psnr"]) for row in rows]})
        except (ValueError, KeyError, OSError):
            st.caption("Les métriques seront affichées après la prochaine sauvegarde.")
    for filename, label in (("model_bundle.pt", "Télécharger le modèle portable"), ("model_export.zip", "Télécharger l’expérience")):
        file = directory / filename
        if file.is_file() and file.stat().st_size < 64 * 1024 * 1024:
            st.download_button(label, file.read_bytes(), file.name, "application/octet-stream", key=f"download_{filename}_{directory.name}")
    audit = read_json(directory / "audit" / "summary.json")
    if audit:
        st.write({"Images décodées": audit["decoded_images"], "Fichiers en erreur": audit["errors_count"],
                  "Groupes de doublons exacts": audit["exact_pixel_duplicate_groups"],
                  "Candidats similaires": audit["near_duplicate_candidates"]})
        st.download_button("Télécharger le rapport d’audit", (directory / "audit" / "summary.json").read_bytes(),
                           "audit_summary.json", "application/json", key=f"audit_{directory.name}")
        sample = directory / "audit" / "sample.jpg"
        if sample.is_file():
            st.image(str(sample), caption="Échantillon à examiner visuellement", width="stretch")
    dataset_report = read_json(directory / "dataset_report.json")
    if dataset_report:
        st.write({"Images vérifiées": dataset_report["valid_images"],
                  "Fichiers installés": dataset_report["installed_files"],
                  "Dossier": dataset_report["destination"]})
        if job and job.get("kind") == "train" and job.get("run_dir") == str(directory) and running:
            st.caption("Données prêtes pour l’entraînement en cours.")
        else:
            st.caption("Ce dossier est prêt pour l’entraînement.")
    with st.expander("Journaux du calcul"):
        st.code(tail_log(directory) or "Aucun journal disponible.", language="text")


loaded_model = weights_panel()
st.title("Tes photos, leur meilleur détail")
st.caption("Débruitage par IA · original préservé · traitement local sur CPU")
photo_tab, data_tab, train_tab, audit_tab = st.tabs(["Corriger une photo", "Préparer les données", "Entraîner le modèle", "Vérifier les données"])
with photo_tab:
    photo_panel(loaded_model)
with data_tab:
    data_panel()
with train_tab:
    training_panel()
with audit_tab:
    audit_panel()
monitor()
