import os, os.path
import csv
import numpy as np
from sklearn.model_selection import train_test_split

def count_images():
    DIR1 = './dataset/flor1'
    DIR2 = './dataset/flor2'
    DIR3 = './dataset/flor3'
    DIR4 = './dataset/flor4'
    DIR5 = './dataset/flor5'
    n_flor1 =len([name for name in os.listdir(DIR1) if os.path.isfile(os.path.join(DIR1, name))])
    n_flor2 =len([name for name in os.listdir(DIR2) if os.path.isfile(os.path.join(DIR2, name))])
    n_flor3 =len([name for name in os.listdir(DIR3) if os.path.isfile(os.path.join(DIR3, name))])
    n_flor4 =len([name for name in os.listdir(DIR4) if os.path.isfile(os.path.join(DIR4, name))])
    n_flor5 =len([name for name in os.listdir(DIR5) if os.path.isfile(os.path.join(DIR5, name))])
    print('Flor 1: %d' %n_flor1 )
    print('Flor 2: %d' %n_flor2 )
    print('Flor 3: %d' %n_flor3 )
    print('Flor 4: %d' %n_flor4 )
    print('Flor 5: %d' %n_flor5 )
    print('Total = %d' %(n_flor1 + n_flor2 + n_flor3 + n_flor4+ n_flor5))

    return([n_flor1, n_flor2, n_flor3, n_flor4, n_flor5])


# Load Data in '.csv' format: filename, class, learning_validation_test_set_flag
def make_csv_file(n_imagens):
    nomes_flores = ['flor1', 'flor2', 'flor3', 'flor4', 'flor5']

    with open('anotations.csv', mode='w') as csv_file:
        csv_file = csv.writer(csv_file, delimiter=',')

        for i in range(len(nomes_flores)):
            for j in range(n_imagens[i]):
                csv_file.writerow( [str(nomes_flores[i])+'_'+str(j)+'.jpg', i+1, 0] )


#make_csv_file(count_images())


def split_dataset(n_imagens):
    nomes_flores = ['flor1', 'flor2', 'flor3', 'flor4', 'flor5']

    x = []

    for i in range(len(nomes_flores)):
        for j in range(n_imagens[i]):
            x.append([str(nomes_flores[i])+'_'+str(j)+'.jpg', i+1, -1])

    x = np.asarray(x)
    y = np.arange(len(x))

    # dividir em treino(60%) e teste+validação(40%)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
    print('x_train: %d, x_test: %d, y_train: %d, y_test: %d' %(len(x_train), len(x_test), len(y_train), len(y_test)))

    # dividir teste+validação(40%) em teste(20%) e validação (20%)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    print('x_val: %d, x_test: %d, y_val: %d, y_test: %d' %(len(x_val), len(x_test), len(y_val), len(y_test)))


    # learning/train = 0, validation = 1, test = 2
    for x in x_train:
        x[2] = 0
    
    for x in x_val:
        x[2] = 1

    for x in x_test:
        x[2] = 2

    data = np.concatenate((x_train, x_val, x_test))


    with open('anotations.csv', mode='w') as csv_file:
        csv_file = csv.writer(csv_file, delimiter=',')
        
        for i in data:
            csv_file.writerow( i )


split_dataset(count_images())