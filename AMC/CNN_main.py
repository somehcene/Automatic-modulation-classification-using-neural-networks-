from numpy import loadtxt, arange, sqrt, zeros, argmax
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, GlobalMaxPooling1D, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from matplotlib.pyplot import plot, xlim, xlabel, ylabel, title, legend, figure
from keras.utils import to_categorical
from numpy.random import seed
from DataGen import MatAR
from sklearn.decomposition import PCA 
seed(2)

#Pré-traitement 

type_mod=['8PSK', '16QAM', '64QAM', 'B-FM', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QPSK']
path='C:/Users/pc/Desktop/M2 ACTUAL/IA/TP4 ia/Dataset_Type_Modulation/'

T_AR=MatAR(200, 'Train', type_mod, path)
Tt_AR=MatAR(60, 'Test', type_mod, path)

dataset = loadtxt('DataARTrain.txt', delimiter=',')
datatest=loadtxt('DataARTest.txt', delimiter=',')

# Extraction de l'entrée (X) de la sortie label (y)
Xtr = dataset[:,0:2048]
Xtr= Xtr.reshape(Xtr.shape[0], Xtr.shape[1], 1)
ytr = dataset[:,2048] 

Xtt = datatest[:,0:2048]
Xtt= Xtt.reshape(Xtt.shape[0], Xtt.shape[1], 1)
ytt = datatest[:,2048] 
   

x_train,  x_val, y_train, y_val=train_test_split(Xtr, ytr, test_size=0.25,  random_state=42)

# Formation du modèle du réseau neuronal
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(2048, 1)))
model.add(BatchNormalization())
model.add(MaxPooling1D(4))

model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(32))
model.add(BatchNormalization())

model.add(Conv1D(filters=64, kernel_size=11, padding='same', activation='relu'))
model.add(MaxPooling1D(16))
model.add(BatchNormalization())
model.add(Flatten())

model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.3))  
model.add(Dense(9, activation = 'softmax'))


# Choix de la compilation
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])

# Exécution de l'apprentissage et récupération
#n_iter = 20 # Nombre d'itération. A modifier pour tester la vitesse d'apprentissage
history = model.fit(x_train, y_train , validation_data=(x_val, y_val), epochs=25, batch_size=20)

# Récupération des pertes Train et Val en fonction des itérations
bce_train = history.history['sparse_categorical_accuracy']
bce_val = history.history['val_sparse_categorical_accuracy']


# Evaluation du modèle
loss, accuracy = model.evaluate(x_val, y_val)
print('Perte globale val : %.2f' % loss)
print('Justesse du modèle val (Accuracy) : %.2f' % (accuracy*100))

# Evaluation du modèle avec test
losstt, accuracytt = model.evaluate(Xtt, ytt)
print('Perte globale test: %.2f' % losstt)
print('Justesse du modèle test (Accuracy) : %.2f' % (accuracytt*100))

# Prédictions des classes avec le modèle
y_pred = (model.predict(x_train)> 0.5).astype(int)
y_predVal= (model.predict(x_val) > 0.5).astype(int)
y_predtt= (model.predict(Xtt) > 0.5).astype(int)

# Rapport de classification
#print('Rapport de classification Train:', classification_report(argmax(y_train, axis=1), argmax(y_pred, axis=1)))

print('Rapport de classification Val:\n', classification_report(y_val, argmax(y_predVal, axis=1)))

# Affichage de l'architecture du réseau
print('Architecture du réseau :')
model.summary()

# Accuracy pour Train et Val
figure()
it = arange(1,len(bce_train)+1)
plot(it, bce_train, label='Train')
plot(it, bce_val, label='Val')
xlim(1,len(it))
xlabel('Itération', fontsize=10)
ylabel('Perte', fontsize=10)
title('Taux de reconnaissance (accuracy)', fontsize=10)
legend()
loss_train = history.history['loss']
loss_val = history.history['val_loss']

figure()
# Courbes des pertes pour train et val 
it = arange(1,len(loss_train)+1)
plot(it, loss_train, label='Train')
plot(it, loss_val, label='Val')
xlim(1,len(it))
xlabel('Itération', fontsize=10)
ylabel('Perte', fontsize=10)
title('Entropie croisée binaire', fontsize=10)
legend()

ConfusionMatrixDisplay(confusion_matrix(y_train, argmax(y_pred, axis=1))).plot()
ConfusionMatrixDisplay(confusion_matrix(y_val, argmax(y_predVal, axis=1))).plot()
ConfusionMatrixDisplay(confusion_matrix(ytt, argmax(y_predtt, axis=1))).plot()

