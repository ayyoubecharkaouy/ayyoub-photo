# AyyPhoto

Projet d'application Android pour améliorer les photos par intelligence
artificielle. Le développement commence par un modèle de débruitage PyTorch,
entraînable sur Google Colab depuis VS Code.

**État actuel :** notebooks d'entraînement et d'évaluation préparés. Les résultats
du modèle restent à mesurer ; l'application Android et l'API ne sont pas encore
implémentées.

## Choisir un notebook

| Notebook | Utilisation |
|---|---|
| [01 — téléchargement KaggleHub](ml/notebooks/01_ayyphoto_denoising.ipynb) | Télécharge DIV2K, demande l'accès à Drive et y conserve le dataset. |
| [01 — dataset déjà sur Drive](ml/notebooks/01_ayyphoto_denoising_drive.ipynb) | Lit directement le dataset existant, sans le télécharger ni le recopier. |

Les deux variantes utilisent le même CNN résiduel, la même configuration et les
mêmes évaluations. Elles sont fournies sans sorties d'exécution ni poids entraînés.

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

