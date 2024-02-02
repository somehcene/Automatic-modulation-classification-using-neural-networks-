
from numpy import loadtxt, arange, zeros, column_stack, savetxt, concatenate 
from matplotlib.pyplot import plot, xlim, xlabel, ylabel, title, legend
from sklearn.preprocessing import normalize

def MatAR(signaux, tr_or_tst, type_mod, path):
    
    num_classes=len(type_mod) #Nombre de classes
    num_sig=signaux    #Nombre de signaux par classe
    num_ech=1024
    #caracteritique des parties réelle et imaginaire de chaque sig

    F=zeros((num_sig*num_classes, num_ech*2))
    Y=zeros((num_sig*num_classes))

    d=0
    for j in range(num_classes):
        for i in range(num_sig):
            if tr_or_tst=='Train':
                i_d=i+1
            if tr_or_tst=='Test':
                i_d=i+202
                
            file_name_train=path+type_mod[j]+'/'+tr_or_tst+'/Trame'+type_mod[j]+f'{i_d:03d}'+'.txt'
            x_i=1024*[0]
            x_Q=1024*[0]

            x=loadtxt(file_name_train, delimiter=",", dtype=complex)
            x_i=x.real
            x_q=x.imag
     
            F[d,:]=concatenate((x_i, x_q))
            Y[d]=int(j)
            d+=1

    #sauvegarde de F et Y 
    f=column_stack([F[:,], Y])
    savetxt('DataAR'+tr_or_tst+'.txt', f, delimiter=',')
    
    return f
