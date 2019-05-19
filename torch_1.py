import torch

Corpus = [
   '<BOS> el puto perro me mordio la puta mano <EOS>',
   '<BOS> la puta anciana me tiro un balazo a los putos pies <EOS>',
   '<BOS> no mames wey chinga tu puta madre <EOS>',
   '<BOS> porque no te vas a la mierda pinche compa alv <EOS>',
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
rnnDim = 100

## Embedding
C = torch.randn(d, N)
### Parametros capa 1
Whh = torch.randn(rnnDim, rnnDim)
Wxh = torch.randn(rnnDim, d)
b = torch.randn(rnnDim)
### Parametros capa 2
W = torch.randn(N-1, rnnDim)
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

def dTanh(xt):
   return 1 - torch.tanh(xt)**2

def forward(xt):
   xt = E[:, xt]
   at = torch.matmul(Whh, ht) + torch.matmul(Wxh, xt) + b
   ht = torch.tanh(at)
   ot = torch.matmul(V, ht) + c
   yt = torch.exp(ot - torch.max(ot))/ torch.sum(ot - torch.max(ot))
   return yt 

def backprop(dt, lr, df):
   gradL_a = torch.matmul(torch.diag(df()))
   dV = torch.ger(dt, ht)
   dc = dt
   dWhh = 
   dWxh = 
   db =
   dE



class rnn():
   def __init__(self, rnnDim, embeddingDim, sigmaSize, activation, weights):
      if weights:
         pass
      else:
         self.Whh = torch.randn(rnnDim, rnnDim)
         self.Wxh = torch.randn(rnnDim, embeddingDim)
         self.b = torch.randn(rnnDim)
         self.V = torch.randn(sigmaSize-1, rnnDim)
         self.c = torch.zeros(sigmaSize-1)
         self.E = torch.randn(embeddingDim, sigmaSize)

   def forward(self, xt):
      xt = self.E[:, xt]
      at = torch.matmul(self.Whh, ht) + torch.matmul(self.Wxh, xt) + self.b
      ht = torch.tanh(at)
      ot = torch.matmul(self.V, ht) + self.c
      yt = torch.exp(ot - torch.max(ot))/ torch.sum(ot - torch.max(ot))
      return yt 

   def backprop(self, delta):
      
      return gradients

   def fit(self, x, y, lr, epochs):
      for epoch in epochs:
         for wi, wj in zip(x, y):
            y_hat = self.forward(wi)
            delta = y_hat[wj] - 1
            gradients = self.backprop()

   
   

#torch.matmul(Whh, ht) + torch.matmul(Wxh, x)

### sueltalo <-> b-side 