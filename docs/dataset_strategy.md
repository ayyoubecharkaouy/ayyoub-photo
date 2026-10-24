# AyyPhoto — vérifier et compléter les données

## Conclusion provisoire

Conserver DIV2K comme première base d'apprentissage et d'évaluation synthétique.
Pour améliorer les photos de smartphones, la priorité est de vérifier les
résultats sur de vraies photos, puis d'ajouter des données représentant les
défauts réellement rencontrés. Ajouter davantage d'images propres du même type
ne garantit pas que le modèle apprendra le bruit réel d'une caméra.

La décision définitive dépend de l'audit du dataset et des résultats du modèle.
Il n'est pas nécessaire de remplacer immédiatement toute la base ou de viser un
nombre arbitraire de milliers d'images.

## Ce qui a réellement pu être vérifié

Le 18 octobre, une inspection locale des en-têtes avait trouvé 900 PNG RGB,
dont 800 images d'entraînement DIV2K et 100 de validation publique. Elle n'était
pas une vérification complète des pixels ni des doublons.

Lors de ce nouvel examen, les photos et leur archive ne sont plus présentes
dans le dossier local `data`. Aucun nouveau résultat d'intégrité, de doublons
ou de diversité visuelle n'a donc été mesuré sur le dataset utilisateur. La
session Kaggle n'est pas accessible depuis l'environnement de travail local.

Un audit autonome est maintenant disponible dans
[03_ayyphoto_data_audit_kaggle.ipynb](../ml/notebooks/03_ayyphoto_data_audit_kaggle.ipynb).
Il doit être exécuté sur Kaggle où les images sont accessibles. Ses tests locaux
utilisent uniquement de petits fichiers artificiels ; ils ne constituent pas
un audit des 900 photos.

## Ce que le pipeline apprend actuellement

Le modèle reçoit une image propre recadrée, transformée et perturbée par un
bruit gaussien. L'image propre sert de cible. Le chargeur ne lit pas de paires
de captures réelles bruitées et propres provenant d'une caméra de smartphone.

Le jeu d'entraînement par défaut contient 720 images sources. Huit recadrages
par image représentent 5 760 exemples générés par époque. Les recadrages et
augmentations apportent de la variété, mais ne créent pas autant de scènes
indépendantes supplémentaires.

Cette expérience permet de vérifier un apprentissage de débruitage mesurable.
Elle ne couvre pas à elle seule la dépendance du bruit à la luminosité,
les traitements internes du téléphone, les artefacts JPEG ou toutes les
conditions de prise de vue. Leur importance réelle doit être vérifiée sur
les appareils et photos visés par le produit.

## Comment décider après l'audit

| Constat | Action utile |
|---|---|
| Fichiers corrompus, transparence ou images trop petites pour le protocole choisi | Revoir ces fichiers avant l'entraînement ; ne pas supprimer automatiquement des images sur la base d'une simple statistique. |
| Doublons ou scènes similaires entre entraînement et validation/test | Regrouper les variantes dans le même ensemble et versionner la nouvelle répartition avant les expériences suivantes. |
| Bons scores synthétiques, mais mauvais rendu sur les photos de téléphone | Prioriser des paires réelles alignées et un jeu d'évaluation adapté aux caméras. |
| Certaines scènes ou conditions sont absentes | Ajouter ces catégories ciblées : portraits, texte, intérieur, nuit et textures selon les usages attendus. |
| Scores de validation en hausse avec plus d'images propres, sans dégradation des tests réels | Élargir progressivement la base propre et mesurer la courbe d'apprentissage. |
| Le dataset représente une autre tâche ou les cibles sont inadaptées | Choisir un dataset plus pertinent plutôt que compenser uniquement par son volume. |

Les indicateurs de contours, de luminosité et d'entropie servent à choisir des
images à inspecter. Un ciel ou un mur peut avoir peu de contours ; une scène
de nuit peut contenir beaucoup de pixels sombres sans être une mauvaise donnée.
Les candidats pHash doivent aussi être vérifiés manuellement.

## Données réelles à ajouter en priorité

Pour le débruitage supervisé, une paire doit montrer la même scène, avec une
entrée bruitée et une référence plus propre correctement alignées. Les
recadrages, l'orientation et les couleurs doivent être cohérents. Une photo
différente simplement jugée plus jolie n'est pas une cible valide.

Commencer par un petit jeu indépendant de photos représentatives des
téléphones visés pour l'évaluation visuelle. Sans référence propre alignée,
ne pas les traiter comme de nouvelles paires d'entraînement supervisé et ne
pas présenter leur PSNR/SSIM comme des scores de fidélité.

Ensuite, si les échecs confirment le besoin, constituer ou sélectionner des
paires réelles autorisées. Conserver une partie du jeu synthétique et des
tests représentatifs pour comparer les gains et régressions. Il n'existe pas
de quantité universelle garantissant une bonne généralisation.

## Attention au chargeur actuel

Le mode `dataset_layout = "div2k"` exige précisément les identifiants 0001–0800
et 0801–0900 dans les dossiers prévus. Il ne suffit pas de déposer de nouvelles
photos à côté : elles seront rejetées par ce contrat.

Le mode `scene_folders` permet une base propre différente, avec des groupes de
scènes explicites. Ajouter de vraies paires bruitées/propres demandera en plus
un chargeur de paires et une nouvelle version du manifeste. Ne pas changer le
jeu de test au milieu d'une comparaison de modèles.

## Résultats à transmettre

Après exécution du notebook 03, transmettre `summary.json`, `sample.jpg` et
`review_candidates.jpg`. Si des doublons sont signalés, joindre aussi les
listes de candidats. Les planches contiennent des miniatures : choisir les
éléments à partager.

Le rapport contrôle les images et les splits de la configuration par défaut
(seed 42). Il ne fournit pas automatiquement de labels sémantiques, de preuve
d'absence de scènes communes ou de mesure de fidélité au bruit d'un capteur.
