# AyyPhoto

Projet d'application Android pour améliorer les photos par intelligence
artificielle. Le développement commence par un modèle de débruitage PyTorch,
entraînable localement sur CPU, sur Kaggle ou sur Google Colab.

**État actuel :** interface Streamlit locale, entraînement CPU et notebooks
d'évaluation préparés. L'application Android et l'API restent à développer.
La qualité du modèle sur les photos réelles doit être mesurée.

## Exécution locale avec environnement virtuel

Depuis le terminal Linux Mint, dans le dossier du projet :

```bash
python3 scripts/setup_local.py
python3 scripts/run_local.py
```

Ouvrir `http://127.0.0.1:8501`. L'installation utilise `.venv` et PyTorch CPU,
sans modifier le Python système. L'interface permet de charger les poids
Kaggle et corriger une photo, lancer un entraînement CPU en arrière-plan ou
auditer un dataset local. Les résultats sont conservés dans `experiments/local`.

Pour les données, tout se fait dans l'interface : dans **Entraîner le modèle**,
garder **DIV2K automatique** puis lancer l'entraînement. Les images sont
vérifiées et les fichiers manquants sont téléchargés avant de commencer le
calcul. Si les 900 images sont valides, aucun accès à Kaggle n'est nécessaire.
L'onglet **Préparer les données** permet aussi de les récupérer séparément.
Le téléchargement n'est pas nécessaire pour seulement corriger une photo.

Les poids et les images ne sont pas inclus dans le dépôt. Pour corriger une
photo, importer un modèle déjà entraîné dans l'interface. Pour entraîner sur
tes propres images, choisir **Mon dataset local** et indiquer son dossier.

Voir le [guide local](docs/local_setup.md) pour les données, la reprise, les tests
et les limites de validation. Si le téléchargement des paquets échoue,
l'installation n'est pas terminée ; ne pas confondre la création de `.venv`
avec une application déjà exécutable.

## Choisir un notebook

| Notebook | Utilisation |
|---|---|
| [01 — Kaggle](ml/notebooks/01_ayyphoto_denoising_kaggle.ipynb) | Lit DIV2K attaché via Add Input ; résultats dans `/kaggle/working`, sans montage Drive. |
| [02 — tester des poids sauvegardés](ml/notebooks/02_ayyphoto_inference_kaggle.ipynb) | Charge un modèle ou un export ZIP et corrige une photo sans DIV2K ni entraînement. |
| [03 — audit des données](ml/notebooks/03_ayyphoto_data_audit_kaggle.ipynb) | Vérifie les images, les doublons et les dimensions, puis produit des planches de revue ; CPU uniquement. |
| [01 — téléchargement KaggleHub](ml/notebooks/01_ayyphoto_denoising.ipynb) | Télécharge DIV2K, demande l'accès à Drive et y conserve le dataset. |
| [01 — dataset déjà sur Drive](ml/notebooks/01_ayyphoto_denoising_drive.ipynb) | Lit directement le dataset existant, sans le télécharger ni le recopier. |

Les trois variantes 01 utilisent le même CNN résiduel et le même protocole
d'évaluation. La variante Kaggle ajoute des exports et la récupération des
expériences. Les notebooks sont fournis sans sorties d'exécution ni poids entraînés.

## Démarrage sur Kaggle

1. Importer `ml/notebooks/01_ayyphoto_denoising_kaggle.ipynb` dans un notebook Kaggle.
2. Dans **Add Input**, ajouter `soumikrakshit/div2k-high-resolution-images`.
3. Activer un GPU dans les options de la session et lancer **Run All**.

Les entrées sont détectées sous `/kaggle/input`. Les checkpoints et rapports
sont dans `/kaggle/working/ayyphoto/experiments`. Enregistrer une version avec
ses sorties ou télécharger l'archive `*_export.zip` avant de quitter la session.
Le fichier `model_bundle.pt` et les anciens `best_weights.pth` peuvent être
ouverts dans le notebook 02 pour tester une photo après fermeture de la session.

Le [réexamen du pipeline Kaggle](docs/kaggle_review.md) détaille les corrections,
les vérifications réalisées et les limites encore à mesurer.
La [stratégie de données](docs/dataset_strategy.md) explique quand conserver,
compléter ou changer le dataset, et distingue les vérifications effectuées des
résultats qui restent à mesurer sur Kaggle.

## Démarrage dans VS Code avec Colab

1. Installer les extensions **Jupyter** et **Colab de Google** recommandées par
   le projet.
2. Ouvrir le notebook choisi et sélectionner **Select Kernel → Colab → New Colab
   Server**, puis un GPU disponible.
3. Pour la variante Drive, vérifier que le dataset existe dans
   `/content/drive/MyDrive/AyyPhoto/data/archive` ou adapter `DATA_ROOT`.
4. Lancer **Run All** et autoriser le montage de Drive lorsque demandé.

Les checkpoints et rapports sont sauvegardés dans
`Mon Drive/AyyPhoto/experiments`. Le guide complet est disponible dans
[ml/notebooks/README.md](ml/notebooks/README.md).

## Première expérience

- DIV2K : 720 images d'entraînement, 80 de validation interne et 100 réservées au
  test final interne, issues de la validation publique du dataset.
- Débruitage synthétique à résolution constante, avec PyTorch, AdamW et perte L1.
- Comparaison avec l'entrée bruitée et une méthode OpenCV, mesures PSNR/SSIM et
  exemples avant/après.
- Sauvegarde des meilleurs poids et de l'état de reprise.
- Essai optionnel sur une photo réelle dans la section 11 des notebooks.

Ce premier modèle apprend sur du bruit synthétique ; sa qualité sur les photos
réelles de smartphones reste à valider. Il ne garantit pas la restauration de
détails absents de l'original.

## Données et documentation

Les datasets, leurs archives, les environnements Python, les poids et les secrets
ne font pas partie du dépôt. Voir [data/README.md](data/README.md) pour les
sources, l'organisation des images et les conditions d'utilisation de DIV2K.

Le périmètre produit et la feuille de route figurent dans
[README_AyyPhoto.md](README_AyyPhoto.md).

