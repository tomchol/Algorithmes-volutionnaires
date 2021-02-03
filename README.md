# Algorithmes évolutionnaires :
Projet python sur les algorithmes évolutionnaires

# Principe de l'algorithme :
La but est de résoudre le cas d'étude des lamps :

On a une piece d'une taille variable et on veut couvrir un maximum de sa surface avec des lampes rondes de diamètre fixé. Pour cela on va utiliser le principe des algorithmes évolutionnaires afin de déterminer les positions et le nombre de lampes optimal pour couvrir la surface sans que les lampes se superposent. 

## Comment cela marche ?

Pour atteindre ce résultat, on part de 20 cas différents où une lampe est positionnée aléatoirement dans la piece. On va ensuite à chaque génération (tour de boucle) faire évoluer cette première population afin de créer des nouvelles générations plus performantes. 

Dans ces évolutions on retrouve : 

La mutation qui consiste à modifier un peu la position de certaines lampes ou à modifier le nombre de lampes en en ajoutant/enlevant une almpe.

La reproduction qui commence à choisir deux espèces à l'aide d'un petit tournois (on prend les plus performantes parmis 3 espèces aléatoires par exemples) et à créer une nouvelle espèce dont les positions des lampes sont déterminées par celles des deux parents.

## Conclusion

En répétant cette opération sur un certain nombre de génération on obteint une disposition presque idéale des lampes dans la pièce. 
