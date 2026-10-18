# Données pour le notebook AyyPhoto

## Dataset déjà enregistré sur Drive

Utiliser [01_ayyphoto_denoising_drive.ipynb](../ml/notebooks/01_ayyphoto_denoising_drive.ipynb)
pour lire directement `Mon Drive/AyyPhoto/data/archive`. Cette variante monte
Drive, audite les images et entraîne le modèle sans télécharger ni recopier
le dataset. Adapter `DATA_ROOT` si le dossier existant se trouve ailleurs.
Les checkpoints et rapports restent sauvegardés dans `Mon Drive/AyyPhoto/experiments`.

## Téléchargement automatique sur Colab et sauvegarde Drive

Sur un noyau Google Colab, y compris depuis VS Code, le notebook installe
KaggleHub et exécute le téléchargement demandé :

```python
import kagglehub

path = kagglehub.dataset_download("soumikrakshit/div2k-high-resolution-images")
print("Path to dataset files:", path)
```

La cellule suivante demande l'autorisation de monter Google Drive. Après
autorisation, le dataset est copié dans `MyDrive/AyyPhoto/data/archive`.
Les fichiers sont vérifiés par SHA-256 ; un fichier identique est réutilisé
et un fichier différent déjà présent n'est pas écrasé. Une copie interrompue
peut être relancée.

La provenance et les empreintes sont enregistrées dans
`MyDrive/AyyPhoto/data/div2k_kaggle_source.json`. Le cache retourné par KaggleHub
sert ensuite aux lectures d'entraînement sur cette session ; le dataset et les
checkpoints sont conservés sur Drive. Les images du PC restent inchangées et
n'ont pas besoin d'être envoyées manuellement.

Sur un noyau local, le téléchargement et le montage Drive sont désactivés par
défaut ; les images déjà présentes dans `data/archive` sont utilisées.

Documentation : [KaggleHub](https://github.com/Kaggle/kagglehub#download-dataset).

## Dataset présent : DIV2K

Le dossier `data/archive` contient 900 images PNG RGB, réparties ainsi :

```text
data/archive/
  DIV2K_train_HR/DIV2K_train_HR/0001.png ... 0800.png
  DIV2K_valid_HR/DIV2K_valid_HR/0801.png ... 0900.png
```

Le notebook utilise directement cette organisation, y compris les dossiers
imbriqués. Ne pas déplacer ni renommer les images. Les noms des dossiers et les
identifiants correspondent à la structure publiée de
[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

Sur le PC, le chemin par défaut est `PROJECT_ROOT / "data" / "archive"` ; sur
Colab, c'est le cache du téléchargement KaggleHub. Le format reste
`dataset_layout = "div2k"`. Avec la seed 42, le notebook réserve 80 des 800
images d'entraînement officielles à la validation interne :

| Usage AyyPhoto | Nombre | Provenance |
|---|---:|---|
| Entraînement | 720 | Entraînement officiel DIV2K |
| Validation interne | 80 | Entraînement officiel DIV2K |
| Test final interne | 100 | Validation publique DIV2K |

Le test interne **n'est pas le test officiel DIV2K**. Il reste désactivé pendant
le choix du modèle. La séparation est faite par image avant les recadrages et
la génération de bruit. DIV2K ne fournit pas ici de labels de scène : les groupes
du manifeste correspondent aux identifiants d'image, et les scènes similaires
ou quasi-doublons doivent être vérifiés manuellement.

L'inspection locale des en-têtes des 900 fichiers n'a rencontré aucune erreur.
Tous sont RGB et assez grands pour les recadrages 256×256 ; taille totale
d'environ 3,71 Gio. Cette inspection ne constitue pas une vérification complète
des pixels ni une recherche de doublons. L'audit complet du notebook s'exécutera
avant tout entraînement lorsque tu lanceras ses cellules.

Les conditions de la source officielle réservent la mise à disposition de DIV2K
à la recherche académique. Conserver l'attribution aux auteurs et clarifier les
droits avant une réutilisation commerciale des images ou du modèle entraîné.
Le notebook télécharge les images depuis Kaggle lors de son exécution sur Colab
et les sauvegarde sur le Drive autorisé. Le dossier local `data/archive/` est
exclu de Git.

Les chemins Colab sont configurés par les cellules de téléchargement et de
sauvegarde. Pour utiliser seulement une copie déjà disponible sur Drive,
désactiver `DOWNLOAD_FROM_KAGGLE` et conserver le montage Drive activé.

## Autre dataset : dossiers de scènes

Choisir `dataset_layout = "scene_folders"`, adapter `dataset_version` et pointer
`DATA_ROOT` vers `data/hq`. Fournir des images propres autorisées ainsi :

```text
data/hq/
  train/
    scene_001/photo_01.png
    scene_001/photo_02.png
    scene_002/photo_01.jpg
  val/
    scene_101/photo_01.png
  test/
    scene_201/photo_01.png
```

Chaque sous-dossier immédiat est un identifiant de scène ou de session. Il doit
être unique entre les trois ensembles. Mettre toutes les prises similaires,
rafales et variantes d'une scène dans le même ensemble. Répartir les scènes,
par exemple autour de 80 % / 10 % / 10 %, avant les expériences. Les noms de
scènes sont attribués manuellement : le notebook ne peut pas déceler à lui seul
que deux dossiers différents montrent le même sujet.

Prévoir au moins une image par ensemble pour vérifier le pipeline. Ce minimum
ne permet pas de conclure à une bonne qualité du modèle. Les images doivent être
RGB ou convertibles en RGB, au format JPEG ou PNG et avoir un petit côté d'au
moins `eval_size` pixels (256 par défaut). Conserver les informations de source
et de licence séparément ; ne pas utiliser de photos personnelles sans accord.

Le notebook vérifie les images, les scènes et les doublons de pixels décodés,
puis écrit un manifeste dans le dossier de l'expérience. Les quasi-doublons
et les erreurs de regroupement nécessitent une revue manuelle.

Le bruit gaussien est généré pendant l'entraînement. Les images sources ne sont
pas modifiées. Cette première expérience ne simule pas complètement le bruit
réel d'un capteur et son traitement JPEG.

Dans Colab, reproduire cette structure sur le stockage de la session ou sur
Google Drive et modifier `DATA_ROOT`. Le dossier `test` reste réservé à
l'évaluation finale, désactivée par défaut.
