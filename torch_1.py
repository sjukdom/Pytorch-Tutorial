import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
torch.manual_seed(123)

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
      if word != '<BOS>' and word != '<EOS>':
         if word not in Sigma:
            Sigma.append(word)

Sigma.append('<EOS>')
Sigma.append('<BOS>')

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
def getBigrams(sentences):
   bigrams = []
   for sentence in sentences:
      bigrams.append([(wi, wj) for wi, wj in zip(sentence[:-1], sentence[1:])])
   return bigrams

bigrams = getBigrams(sentIndexed)

print('\n')
for sentence in bigrams:
   print(sentence)

# Imprimir bigramas de la primera oracion
print('\n')
for pair in bigrams[0]:
   print('({}, {})'.format(indx2word[pair[0]], indx2word[pair[1]]))


# Red neuronal recurrente
class RNN():
   def __init__(self, rnnDim, embeddingDim, sigmaSize, weights):
      if weights:
         pass
      else:
         self.Whh = torch.randn(rnnDim, rnnDim) / np.sqrt(rnnDim)
         self.Wxh = torch.randn(rnnDim, embeddingDim) / np.sqrt(embeddingDim)
         self.b = torch.zeros(rnnDim)
         self.V = torch.randn(sigmaSize-1, rnnDim) / np.sqrt(rnnDim)
         self.c = torch.zeros(sigmaSize-1) 
         self.E = torch.randn(embeddingDim, sigmaSize) / np.sqrt(sigmaSize)
         self.dim = rnnDim

   def softmax(self, x):
      xnew = torch.exp(x - torch.max(x))
      return xnew/torch.sum(xnew)

   def fit(self, bigrams, lr, epochs):
      Loss = []
      loss = []
      for epoch in range(epochs):
         Loss_per_epoch = 0
         for i, sentence in enumerate(bigrams):
            #print('\nSentence: ', i)
            # Separar los bigramas en 2 listas
            x = [bigram[0] for bigram in sentence]
            y = [bigram[1] for bigram in sentence]
            # Lista del tiempo 
            t = range(1, len(sentence) + 1)
            # Inicializar el estado oculto en ceros
            ht = torch.zeros(len(sentence)+1, self.dim)
            for wi, wj, t in zip(x, y, t):
               #print('t[{}] x={}, y={}'.format(t, indx2word[wi], indx2word[wj]))
               # Forward
               xt = self.E[:, wi]
               at = torch.matmul(self.Whh, ht[t-1]) + torch.matmul(self.Wxh, xt) + self.b
               ht[t] = torch.tanh(at)
               ot = torch.matmul(self.V, ht[t]) + self.c
               yt = self.softmax(ot)
               # Error
               yti = torch.zeros(yt.shape)
               yti[wj] = 1
               dt = yt - yti
               # Loss 
               loss.append(-torch.log(yt[wj]))     
               #print(yt[wj])  
               # Backprop
               gradL_a = torch.matmul(torch.diag(1-torch.tanh(at)**2), torch.matmul(torch.t(self.V), dt))
               dV = torch.ger(dt, ht[t])
               dc = dt
               dWhh = torch.ger(gradL_a, ht[t-1])
               dWxh =torch.ger(gradL_a, xt) 
               db = gradL_a
               dE =  torch.matmul(torch.t(self.Wxh), gradL_a)
               # Gradient descent
               self.V -= lr*dV
               self.c -= lr*dc
               self.Whh -= lr*dWhh
               self.Wxh -= lr*dWxh
               self.b -= lr*db
               self.E[:, wi] -= lr*dE
            Loss_per_epoch += np.sum(loss)
            print('Loss E{} Sent{} = {}'.format(epoch+1, i, np.sum(loss)))
            loss.clear()
         Loss.append(Loss_per_epoch)
         Loss_per_epoch = 0
      return Loss
   
rnn = RNN(100, 200, len(Sigma), None)
loss = rnn.fit(bigrams, 0.1, 100)

plt.plot(loss, 'r')
plt.title('Negative log-likelyhood')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
### sueltalo <-> b-side 