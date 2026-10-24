# AyyPhoto sur Linux Mint : environnement virtuel et Streamlit

Cette version fonctionne sur CPU, sans montage Google Drive. Elle
prévoit le ThinkPad X270 vérifié dans cette session : i5-7200U, 8 Go de RAM et
Intel HD 620. Aucune bibliothèque CUDA n'est requise.

## Installation

Depuis la racine du projet, avec Python 3.10 ou plus récent :

```bash
python3 scripts/setup_local.py
python3 scripts/run_local.py
```

Le premier script crée ou complète `.venv`, puis installe PyTorch depuis l'index
CPU officiel et les dépendances de `requirements.txt`. Il ne modifie pas le
Python système. Une connexion Internet est nécessaire pour cette installation.
Le second script utilise obligatoirement le Python de `.venv`, quelle que soit
la sélection de noyau de VS Code, puis ouvre un serveur local sur
`http://127.0.0.1:8501`. Ouvrir cette adresse dans le navigateur.

Pour vérifier l'environnement ou travailler directement avec celui-ci :

```bash
python3 scripts/setup_local.py --check
source .venv/bin/activate
python -m streamlit run app.py
```

Le lancement doit être effectué depuis la racine pour les commandes directes.
Les scripts de lancement résolvent eux-mêmes le dossier du projet.

## Télécharger les données une seule fois

Dans **Entraîner le modèle**, garder **DIV2K automatique** puis cliquer sur
**Lancer l'entraînement CPU**. L'application prépare
`soumikrakshit/div2k-high-resolution-images` dans `data/archive`, puis démarre
l'entraînement. Le dossier peut être absent au départ.

Pour préparer les images sans entraîner, ouvrir **Préparer les données**,
choisir le dossier puis cliquer sur **Préparer les données DIV2K**. Le bouton
**Vérifier la présence** indique les fichiers repérés ; la préparation vérifie
ensuite leur intégrité complète.

- Il vérifie les identifiants attendus, les dimensions, le décodage et les
  contrôles PNG des images déjà présentes.
- Si les 900 images sont valides, il ne contacte pas Kaggle et ne les recopie pas.
- Pour quelques images absentes ou abîmées, il demande les fichiers concernés.
  Pour une installation initiale, il récupère le dataset complet via KaggleHub.
- Les images valides sont conservées. Un fichier corrompu est sauvegardé avec
  un suffixe `.invalid_<date>` avant son remplacement vérifié.
- Une relance reprend la préparation à partir des fichiers valides déjà installés.

Le cache KaggleHub se trouve par défaut dans `.cache/kagglehub`, séparément du
dataset final. Le premier téléchargement nécessite donc de la place pour le
cache et les images finales, soit plusieurs Gio. Ces dossiers et les images
ne sont pas ajoutés à Git. La validation n'est pas une preuve de diversité des
scènes ; l'audit du dataset reste disponible pour ce travail.

Si la préparation échoue, l'erreur apparaît dans le suivi et l'entraînement
ne commence pas. Après correction du problème, relancer depuis l'interface.
Le chargeur répartit ensuite les
images propres en 720 train / 80 validation / 100 test interne et génère le
bruit lors de l'entraînement.

La préparation s'exécute en arrière-plan avec son journal et sa progression.
Un dataset déjà vérifié localement reste utilisable sans connexion réseau.
Le fichier [scripts/download_data.py](../scripts/download_data.py) reste
disponible pour les automatisations ; aucune commande de données n'est requise
pour utiliser l'interface.

## Corriger une photo

1. Dans la colonne de gauche, importer `best_weights.pth`, `model_bundle.pt`,
   `last.ckpt` ou un ZIP d'export AyyPhoto. Les anciens exports Kaggle sont
   acceptés. Les poids peuvent aussi être déposés dans `artifacts/` et choisis
   dans la liste des modèles locaux.
2. Importer une photo JPEG ou PNG dans l'onglet **Corriger une photo**.
3. Choisir l'intensité et cliquer **Corriger la photo**.
4. Comparer les deux aperçus, puis télécharger le PNG corrigé.

Le traitement est local, par tuiles sur CPU. Il garde les dimensions de l'image
après orientation EXIF et n'écrase jamais l'original. L'export sRGB ne recopie
pas les métadonnées EXIF. Les limites de 32 Mio par photo, 16 mégapixels et
64 Mio par modèle bornent la mémoire utilisée sur ce PC.

Aucun modèle entraîné n'est fourni dans Git. Si les anciens poids Kaggle ont
été perdus sans sauvegarde, l'interface ne peut pas les reconstruire. Il faut
retrouver un checkpoint ou entraîner une nouvelle version avec des données.

## Entraîner sur CPU

Dans **Entraîner le modèle**, choisir **DIV2K automatique** pour préparer les
images avant le calcul, ou **Mon dataset local** pour utiliser tes propres
images sans téléchargement. Deux organisations sont acceptées :

```text
data/archive/
  DIV2K_train_HR/DIV2K_train_HR/0001.png ... 0800.png
  DIV2K_valid_HR/DIV2K_valid_HR/0801.png ... 0900.png
```

ou une base personnelle organisée par scènes indépendantes :

```text
data/hq/
  train/scene_001/photo.png
  train/scene_002/photo.png
  val/scene_101/photo.png
  test/scene_201/photo.png   # optionnel pour le calcul local actuel
```

Le profil **Essai rapide** utilise au maximum 40 images train, 8 images de
validation, 2 époques, des recadrages 64×64, un batch de 1 et 2 threads CPU.
Ce profil teste le parcours ; il ne valide pas la qualité finale. Le profil
complet utilise toutes les images, les recadrages 128×128 et une validation
sur des recadrages 256×256. La durée doit être mesurée sur le PC.

Les scores des deux profils ne sont pas directement comparables puisque la
taille et le nombre de recadrages évalués diffèrent. Les paramètres et le
protocole sont conservés avec chaque expérience. Aucun test final n'est utilisé
automatiquement pour régler le modèle.

Le calcul est un processus séparé avec le Python de `.venv`. Les journaux,
pertes et scores sont visibles dans **Tes expériences locales**. Le bouton
d'arrêt demande une interruption ; `last.ckpt` représente toujours une époque
complète. Les poids interrompus sont réservés à l'inférence ou au fine-tuning.

Les fichiers sont conservés dans `experiments/local/train_<date>/` :

- `best_weights.pth` et `model_bundle.pt` pour l'inférence ;
- `last.ckpt` pour une reprise locale avec optimiseur et états aléatoires ;
- `history.csv`, `summary.json`, `manifest.json` et `config.json` ;
- `model_export.zip` pour conserver ou transférer l'expérience.

Pour commencer un nouvel entraînement à partir de poids Kaggle, renseigner
**Poids initiaux**. Pour reprendre un entraînement local à la dernière époque
complète, renseigner **Dossier d'expérience locale à reprendre**, conserver les
réglages et augmenter le nombre total d'époques. Les deux modes sont distincts :
le checkpoint complet Kaggle n'est pas un état de reprise locale identique.

Les originaux de reprise ne sont pas écrasés : chaque calcul crée un nouveau
dossier. Les fichiers locaux restent sur disque après fermeture du navigateur.
Une extinction du PC ou un arrêt du processus nécessite de repartir d'un
checkpoint ; la seule fermeture du navigateur ne constitue pas une sauvegarde.

## Auditer les données

Dans **Vérifier les données**, renseigner le dossier puis lancer l'audit. Les
résultats et planches sont dans `experiments/local/audit_<date>/audit/`.
Les images sources restent inchangées. Les ressemblances signalées et les
indices de contours nécessitent une revue visuelle.

## Validation

Après installation, exécuter :

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s ml/tests -v
```

La suite locale inclut un chargement des anciens formats de poids, la
comparaison tuiles/image complète, un petit entraînement artificiel avec reprise
et un test de démarrage Streamlit. Ces tests ne lancent pas d'entraînement sur
les photos du projet. Sans PyTorch/Streamlit, les contrôles correspondants sont
explicitement ignorés, et non considérés comme validés.

Les 45 tests passent dans `.venv`, y compris le démarrage de l'interface,
le lancement automatique avec un dossier de données absent et le petit
entraînement artificiel avec reprise. Le téléchargement Kaggle réel n'a pas
été validé : KaggleHub reste absent et son installation a échoué dans cet
environnement. Relancer l'installation initiale depuis le terminal Linux Mint
disposant d'un accès aux paquets requis.

Références : [installation PyTorch](https://pytorch.org/get-started/locally/),
[chargement de fichiers Streamlit](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader),
[tests Streamlit](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).
