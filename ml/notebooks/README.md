# Premier modèle AyyPhoto

## Vérifier les images avant d'ajouter des données

Ouvrir [03_ayyphoto_data_audit_kaggle.ipynb](03_ayyphoto_data_audit_kaggle.ipynb)
et attacher le dataset avec **Add Input**. Aucun GPU ni PyTorch n'est requis.
L'audit lit tous les pixels, cherche les doublons exacts et les ressemblances
possibles, contrôle les dimensions et produit des planches de revue.
Il ne supprime et ne modifie aucune image source.

Les sorties sont sous `/kaggle/working/ayyphoto_data_audit`. Conserver
`summary.json`, `sample.jpg`, `review_candidates.jpg` et les listes de doublons
si elles ne sont pas vides. Les ressemblances et statistiques de contours ne
constituent pas des décisions automatiques de suppression.

L'outil peut aussi s'exécuter sur un dossier local d'images avec NumPy et Pillow :

```bash
python3 ml/data_audit.py --data-root /chemin/des/images --output /chemin/du/nouveau-rapport
```

Voir la [stratégie de données](../../docs/dataset_strategy.md) pour interpréter
les résultats et décider s'il faut compléter ou remplacer la base.

## Tester un modèle sauvegardé sans réentraîner

Ouvrir [02_ayyphoto_inference_kaggle.ipynb](02_ayyphoto_inference_kaggle.ipynb).
Attacher les sorties de l'entraînement et la photo à tester avec **Add Input**,
puis renseigner `MODEL_PATH` et `PHOTO_PATH`. Une seule expérience peut être
détectée automatiquement ; plusieurs expériences exigent un choix explicite.
Le notebook accepte `model_bundle.pt`, les anciens `best_weights.pth`, `last.ckpt`
et les archives `*_export.zip`. Il n'a besoin ni de DIV2K ni des cellules
d'entraînement. Les images et rapports sont écrits dans
`/kaggle/working/ayyphoto_tests`.

## Variante Kaggle — sans Google Drive

Si le calcul s'effectue sur Kaggle, ouvrir
[01_ayyphoto_denoising_kaggle.ipynb](01_ayyphoto_denoising_kaggle.ipynb).
Les variantes Colab ne doivent pas être utilisées sur Kaggle : la présence
de `google.colab` parmi les bibliothèques n'y rend pas `drive.mount()` disponible.

1. Importer cette variante dans Kaggle.
2. Dans **Add Input**, attacher `soumikrakshit/div2k-high-resolution-images`.
3. Sélectionner un GPU dans les options du notebook.
4. Cliquer **Run All**. Le dossier DIV2K est détecté sous `/kaggle/input`, même
   s'il se trouve dans une arborescence imbriquée. S'il n'est pas trouvé,
   KaggleHub récupère le chemin de `soumikrakshit/div2k-high-resolution-images`.
   En cas de plusieurs copies,
   renseigner `DATA_ROOT` dans la configuration.

Cette variante utilise les entrées Kaggle ou le chemin fourni par KaggleHub,
sans copie du dataset ni montage Drive.
`INSTALL_DEPENDENCIES = False` conserve les bibliothèques préinstallées ;
activer cette option et Internet uniquement si une dépendance manque.
L'entraînement est activé et le test final reste désactivé.

Les résultats sont écrits dans `/kaggle/working/ayyphoto/experiments`.
Un ZIP contenant les poids, la configuration et les métriques est actualisé
après chaque époque complète par défaut. `model_bundle.pt` permet l'inférence
sans le dossier d'expérience complet. Enregistrer une version avec ses sorties
ou télécharger l'archive avant de quitter : ces exports ne rendent pas la
session persistante à eux seuls.

Pour reprendre un checkpoint ajouté comme entrée Kaggle, renseigner
`RESUME_RUN_DIR` avec le dossier d'expérience contenant `config.json`,
`best_weights.pth` et `last.ckpt`. Les artefacts sont copiés automatiquement
vers un nouveau dossier de sortie accessible en écriture. Le dataset n'est
pas recopié. Mettre `RUN_TRAINING = False` pour charger sans continuer
l'entraînement. Une interruption volontaire garde `interrupted_weights.pth`
pour l'inférence ou un futur fine-tuning ; la reprise de l'état d'entraînement
se fait avec `last.ckpt`, à la dernière époque complète.

`RGB_CACHE_MIB` borne un cache RGB uint8 pour éviter certains redécodages. Les
recadrages sont convertis en flottants après leur extraction. Les paires et
bruits restent identiques à l'ancien pipeline. La validation journalise la L1
brute et écrêtée séparément. Les tests finaux sont associés aux empreintes des
poids et du protocole ; un ancien rapport sans ces empreintes n'est pas
automatiquement attribué au modèle courant.

Voir l'[analyse du pipeline](../../docs/kaggle_review.md) pour les limites de
validation et les prochaines expériences.

Voir la [documentation Kaggle](https://www.kaggle.com/docs/notebooks).

## Variante pour un dataset déjà présent sur Drive

Ouvrir [01_ayyphoto_denoising_drive.ipynb](01_ayyphoto_denoising_drive.ipynb)
si DIV2K est déjà enregistré sur Drive. Cette variante conserve le modèle,
les paramètres et les évaluations du notebook 01, sans KaggleHub, sans
téléchargement et sans recopie du dataset.

Depuis VS Code, connecter un GPU Colab, vérifier le chemin du dataset dans la
section 2, puis lancer **Run All**. Autoriser le montage de Drive si demandé.
Les chemins par défaut sont :

```python
DATA_ROOT = Path("/content/drive/MyDrive/AyyPhoto/data/archive")
OUTPUT_ROOT = Path("/content/drive/MyDrive/AyyPhoto/experiments")
```

L'entraînement est activé et le test final reste désactivé. Le dataset est lu
directement sur Drive ; seuls les checkpoints et rapports d'expérience y sont
écrits. Les cellules n'ont pas été exécutées par l'agent.

## Notebook 01 avec téléchargement KaggleHub

Ouvrir `01_ayyphoto_denoising.ipynb`. Le notebook est autonome : aucun module
Python du dépôt n'est nécessaire. Il est livré sans sorties et sans exécution.

Sur Colab, le notebook télécharge DIV2K avec KaggleHub puis demande l'accès à
Google Drive pour y enregistrer le dataset. Sur un noyau local, il utilise les
900 images de `data/archive`. Il prépare une répartition reproductible de
720 images pour l'entraînement, 80 pour la validation et 100 pour le test final
interne. Les 100 dernières viennent de la validation publique DIV2K ; ce ne sont
pas les images du test officiel. Aucun déplacement des fichiers n'est requis.

## VS Code sur le PC avec un noyau Google Colab

1. Installer l'extension officielle [Colab de Google](https://marketplace.visualstudio.com/items?itemName=google.colab)
   et l'extension Jupyter dans VS Code. Les extensions sont recommandées dans
   `.vscode/extensions.json`.
2. Ouvrir le notebook local, puis choisir **Select Kernel → Colab → New Colab
   Server**. Sélectionner un GPU disponible et se connecter au compte Google.
3. Cliquer **Run All**. La cellule KaggleHub télécharge automatiquement
   `soumikrakshit/div2k-high-resolution-images` dans le cache du serveur Colab.
4. Après le téléchargement, autoriser le montage de Drive lorsque le notebook
   le demande. La cellule suivante copie et vérifie le dataset dans
   `MyDrive/AyyPhoto/data/archive`. Les fichiers identiques sont réutilisés ;
   les fichiers existants différents ne sont pas écrasés.
5. Une fois la sauvegarde terminée, les cellules vérifient les données et
   CUDA avant l'entraînement. L'installation et l'entraînement sont activés ;
   le test final reste désactivé.

Le transfert manuel des images du PC n'est plus nécessaire. La connexion au
serveur et l'autorisation de Drive restent à effectuer par l'utilisateur.
Le notebook a été vérifié statiquement et sa fonction de copie a été testée sur
des fichiers temporaires ; aucune exécution distante n'a encore été validée.

Voir le [guide officiel de l'extension](https://github.com/googlecolab/colab-vscode/wiki/User-Guide).

## Premier lancement de l'entraînement

L'installation des dépendances et l'entraînement sont maintenant activés dans
le notebook : `INSTALL_DEPENDENCIES = True` et `RUN_TRAINING = True`.
Le test final reste désactivé (`RUN_FINAL_TEST = False`). Aucun entraînement
n'a encore été exécuté par l'agent.

La configuration utilise au maximum 30 époques, des recadrages 128×128 et un
batch de 8. Exécuter les cellules dans l'ordre après choix du noyau et des chemins.
Les fichiers `last.ckpt` et `best_weights.pth` sont sauvegardés pendant
l'entraînement dans le dossier indiqué par `Dossier de l'expérience`.

Sur Colab, `DOWNLOAD_FROM_KAGGLE` et `MOUNT_GOOGLE_DRIVE` sont activés
automatiquement. Après téléchargement et sauvegarde sur Drive, `DATA_ROOT`
utilise le cache KaggleHub du serveur pour l'entraînement. `OUTPUT_ROOT` vise
`/content/drive/MyDrive/AyyPhoto/experiments` pour conserver les checkpoints.
L'ordre est : téléchargement → demande d'accès Drive → copie vérifiée →
audit du dataset → entraînement. Aucun chemin n'a besoin d'être modifié pour
ce premier lancement dans un noyau Colab.

Le fichier `MyDrive/AyyPhoto/data/div2k_kaggle_source.json` conserve le handle
Kaggle, le chemin de version retourné et les empreintes des fichiers copiés.
Une relance vérifie les fichiers déjà sauvegardés avant de les réutiliser.

Choisir un environnement GPU avant l'exécution. Vérifier dans la sortie matériel
que `device` vaut `cuda`. Si le montage ou l'installation demande un redémarrage,
reprendre les cellules depuis le début après ce redémarrage.

Pour la première analyse, transmettre la sortie matériel, le tableau des effectifs
et les lignes de résultats des premières époques. À la fin, transmettre
`summary.json`, `learning_curves.png` et `comparison_validation.png`.

## VS Code

1. Installer Python 3.10 ou plus récent et les extensions Microsoft Python et
   Jupyter.
2. Depuis la racine du projet, préparer un environnement :

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install ipykernel
   ```

3. Ouvrir le notebook et sélectionner le noyau `.venv` avec **Select Kernel**.
4. Si nécessaire, activer `INSTALL_DEPENDENCIES` dans la première cellule de code
   et l'exécuter. Pour CUDA, installer une distribution PyTorch adaptée à sa
   machine depuis [PyTorch](https://pytorch.org/get-started/locally/).
5. Vérifier les images selon [data/README.md](../../data/README.md), ajuster les
   chemins si nécessaire, puis exécuter les cellules dans l'ordre. Mettre `RUN_TRAINING = True`
   seulement au moment de lancer l'entraînement.

Le GPU NVIDIA est utilisé si PyTorch détecte CUDA ; sinon le notebook utilise
le CPU. Il n'installe aucun pilote GPU.

## Google Colab

1. Importer le fichier `.ipynb` dans [Google Colab](https://colab.research.google.com/).
2. Choisir un environnement d'exécution avec GPU, si disponible.
3. Exécuter les cellules dans l'ordre : installation, téléchargement KaggleHub,
   puis autorisation du montage Drive.
4. Attendre la copie et la vérification du dataset sur Drive. Les chemins des
   données et des checkpoints sont configurés automatiquement.
5. L'audit des données et l'entraînement suivent. Si Kaggle exige une connexion,
   exécuter `kagglehub.login()` sans inscrire de secret dans le notebook, puis
   relancer la cellule de téléchargement.

Le fichier fonctionne aussi lorsqu'il est importé seul, sans le dépôt. Le
stockage sous `/content` est temporaire et le GPU n'est pas garanti.

## Résultats à transmettre

Le dossier de l'expérience contiendra notamment :

- `summary.json` : configuration, matériel, meilleure époque et scores ;
- `history.csv` et `learning_curves.png` : progression de l'entraînement ;
- `validation_metrics.csv` et `comparison_validation.png` : comparaisons avec
  l'original bruité et OpenCV ;
- `best_weights.pth` et `last.ckpt` : modèle et état de reprise ;
- `manifest.csv`, `data_audit.json` et `environment.txt` : traçabilité.

Envoyer `summary.json`, les courbes et la comparaison visuelle pour l'analyse.
Il n'est pas nécessaire de transmettre les poids ni les images sources. Si une
erreur survient, copier son message complet et préciser la cellule concernée.

Le test final nécessite `RUN_FINAL_TEST = True`, après choix définitif du
modèle. Ne pas l'utiliser à chaque réglage des hyperparamètres. Aucune version
du modèle n'est automatiquement promue en production.

## Références

- [KaggleHub : téléchargement de datasets](https://github.com/Kaggle/kagglehub#download-dataset)
- [Dataset Kaggle demandé](https://www.kaggle.com/datasets/soumikrakshit/div2k-high-resolution-images)
- [DIV2K : données et conditions d'utilisation](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- [Notebooks Jupyter dans VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
- [FAQ Google Colab](https://research.google.com/colaboratory/faq.html)
- [Sauvegarde et chargement PyTorch](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [Métriques scikit-image](https://scikit-image.org/docs/stable/api/skimage.metrics.html)
