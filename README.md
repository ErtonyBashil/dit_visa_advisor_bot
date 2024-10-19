### Mise en place d'un chatbot sous le systeme RAG
Chatbot pour accompagner les étudiants étrangers à la carte de séjour au Sénégal
<hr>

Ce projet consiste à mettre en place un assistant virtuel pour aider les étrangers, particuliement les étudiants dans leur démarche pour obtenir la carte de séjour.
un chatbot au service des étrangers, un outil intéractif dynamique pour humaniser l'assistance de l'office de l'étranger aux étudiants exemptés d'avoir une 
carte de séjour au Sénegal.

Dans ce projet nous avons utilisé Mistral comme le modèle LLM, Langchain comme framme de gestion des modèles LLM
chromaDB comme une base des données vectorielles.


![Architecture RAG](images/rag_architecture.jpg)

RAG est une architecture hybride qui combine deux concepts clés dans le traitement du langage naturel (NLP) :

* Récupération d'information (retrieval) : Lorsqu'un modèle cherche dans une base de données externe ou un 
ensemble de documents pertinents pour récupérer les informations nécessaires. 


* Génération de texte (generation) : Une tâche dans laquelle un modèle génère du texte de manière fluide et
cohérente en fonction d'une requête ou d'une information donnée. 

En resumé modèle interroge une base de données ou un corpus externe pour trouver des documents ou informations 
pertinents par rapport à une requête et à partir des informations récupérées, le modèle génère une réponse ou
un contenu en langage naturel. ils ont tendance à avoir des hallucinations lorsque le sujet porte sur des 
informations qu'ils ne « connaissent pas », c'est-à-dire qu'elles n'étaient pas incluses dans leurs données 
de formation.




