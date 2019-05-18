import torch

Corpus = [
   '<BOS> el puto perro me mordio la mano <EOS>',
   '<BOS> la puta anciana me tiro un balazo a los pies <EOS>',
   '<BOS> no mames wey chinga tu madre <EOS>',
   '<BOS> porque no te vas a la mierda compa <EOS>',
   '<BOS> haste a un lado pinche perro puto <EOS>',
]

# Crear una lista de palabras por cada oracion
s = [sentence.split() for sentence in Corpus]

# Crear alfabeto del corpus
Sigma = []
for sentence in s:
   for word in sentence:
      if word not in Sigma:
         Sigma.append(word)

print('\nAlfabeto:')
print(Sigma)

# Crear diccionario de palabras a indices del alfabeto
word2indx = {}
indx2word = {}
for indx, word in enumerate(Sigma):
   word2indx[word] = indx
   indx2word[indx] = word

# Representando oraciones mediate indices del alfabeto
sentIndexed = []
for sentence in s:
   sentIndexed.append([word2indx[word] for word in sentence])
   
print('\n')
for sidx in sentIndexed:
   print(sidx)

# Crear bigramas del corpus
bigrams = []
for sentence in sentIndexed:
   for wi, wj in zip(sentence[:-1], sentence[1:]):
      bigrams.append((wi, wj))

print('\n')
print(bigrams)

for pair in bigrams:
   print('({}, {})'.format(indx2word[pair[0]], indx2word[pair[1]]))


# Red neuronal recurrente

## Parametros de la red
N = len(Sigma)
d = 100
cell_dim = 100

## Embedding
C = torch.randn(d, N)
### Parametros capa 1
Whh = torch.randn(cell_dim, cell_dim)
Wxh = torch.randn(cell_dim, d)
b = torch.randn(cell_dim)
### Parametros capa 2
W = torch.randn(N-1, cell_dim)
c = torch.zeros(N-1)

## Estado oculto
ht = torch.zeros(5, 2)

print('\nWhh')
print(Whh)

print('\nWxh')
print(Wxh)

print('\nV')
print(V)

print('\nht0')
print(ht)


def forward(x):
   pass
   

#torch.matmul(Whh, ht) + torch.matmul(Wxh, x)