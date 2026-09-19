from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

sentence1 = "Minimum CGPA required is 6.5"
sentence2 = "What is the minimum CGPA needed?"

embedding1 = model.encode([sentence1])
embedding2 = model.encode([sentence2])

similarity = cosine_similarity(embedding1, embedding2)

print("Similarity score:", similarity[0][0])