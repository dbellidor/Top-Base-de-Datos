import tensorflow as tf
import numpy as np
from tqdm import tqdm
from random import  randint
from numpy import genfromtxt

class Modified_SOM(object):
    """
    Mapa de autoorganizacion 2D con funcion de vecindad gaussiana
    y disminuyendo linealmente la tasa de aprendizaje.
    """

    #Para comprobar si el SOM ha sido entrenado.
    _trained = False

    def __init__(self, m, n, dim, indata,n_iterations=100, alpha=None, sigma=None):
        """
        Inicializa todos los componentes necesarios del TensorFlow Graph.

	m X n son las dimensiones del SOM. 'n_iterations' debe ser un entero
	que denote el número de iteraciones que se han realizado durante 
	el entrenamiento.
	'dim' es la dimensionalidad de las entradas de entrenamiento.
	'alpha' es un numero que denota la velocidad de aprendizaje basada
	en el tiempo inicial (sin iteracion). El valor predeterminado es 0.3
	'sigma' es el valor de vecindad inicial, que denota el radio de 
	influencia de la BMU durante el entrenamiento. Por defecto, se 
	considera que es la mitad del maximo (m, n).
        """

        #Asignar las variables requeridas primero
        self._m = m
        self._n = n
        if alpha is None:
            alpha = 0.3
        else:
            alpha = float(alpha)
        if sigma is None:
            sigma = max(m, n) / 2.0
        else:
            sigma = float(sigma)
        self._n_iterations = abs(int(n_iterations))

        ##INICIALIZAR EL GRAFO
        self._graph = tf.Graph()

        ##GRAFO CON COMPONENTES NECESARIOS
        with self._graph.as_default():

            ##VARIABLES Y OPS CONSTANTES PARA ALMACENAMIENTO DE DATOS

            #Vectores de pesos inicializados aleatoriamente para todas las neuronas,
            #almacenados juntos como una matriz Variable de tamaño [m*n,dim]
            indata=np.array(indata)
            dist_array=np.zeros((indata.shape[0],1))
            for i in range(indata.shape[0]):
                for j in range(indata.shape[0]):
                    dist_array[i] = dist_array[i] + np.sqrt( (indata[i,0]-indata[j,0])*(indata[i,0]-indata[j,0]) + (indata[i,1]-indata[j,1])*(indata[i,1]-indata[j,1]) )
            sum_dist=0.0
            mn_dist=100000.0
            mx_dist=0.0

            for i in range(indata.shape[0]):
                sum_dist = sum_dist + dist_array[i]
                if dist_array[i]>mx_dist:
                    mx_dist=dist_array[i]
                if dist_array[i]<mn_dist:
                    mn_dist=dist_array[i]

            avg_dist=sum_dist/indata.shape[0]

            neuron_dist=np.zeros((m*n,1))
            diff_neuron = (mx_dist-mn_dist) / (m * n)

            uxy=np.zeros((m*n,2))
            for i in range(m*n):
                neuron_dist[i] = mn_dist + diff_neuron*(i-0.5)
                mn_2 = 1000000.0
                mn_index = -1
                for j in range(indata.shape[0]):
                    tmp_dist = np.fabs( neuron_dist[i] - dist_array[j] )
                    if tmp_dist<mn_2:
                        mn_2=tmp_dist
                        mn_index=j
                uxy[i,0] = indata[mn_index,0]
                uxy[i,1] = indata[mn_index,1]

            print(uxy)
            uxy = np.array(uxy)
            print(uxy.dtype)



            #self._weightage_vects = tf.Variable(tf.random_normal([m*n, dim]))
            self._weightage_vects = tf.Variable(tf.convert_to_tensor(uxy,tf.float64,name=None,preferred_dtype=tf.float64))
            print('1: ',np.array(self._weightage_vects))
            for i in range(self._weightage_vects.shape[0]):
                print(np.array(self._weightage_vects.read_value()[i][0])," , ",np.array(self._weightage_vects[i][1]))
            self._weightage_vects=tf.Variable(tf.constant(uxy))
            self._weightage_vects=tf.Variable(tf.cast(self._weightage_vects,tf.float32))
            print('2: ',self._weightage_vects.value())

            print(self._weightage_vects.dtype)

            #Matriz de tamaño [m * n, 2] para ubicaciones de la grid de SOM
            #para las neuronas
            self._location_vects = tf.constant(np.array(
                list(self._neuron_locations(m, n))))

            ##Lugar para la entrada del entrenamiento
            #Necesitamos asignarlos como atributos a uno mismo, ya que
            #sera la entrada durante el entrenamiento

            #El vector de entrenamiento
            self._vect_input = tf.placeholder("float", [dim])
            # Numero de iteracion
            self._iter_input = tf.placeholder("float")

            ##CONSTRUCCION DEL ENTRENAMIENTO PIEZA POR PIEZA
            #Solamente al final, la operacion de entrenamiento'root' necesita ser asignado como
            #un atributo para si mismo, ya que todo el resto se ejecutara automaticamente
            #durante el entrenamiento.

            #Para calcular la Best Matching Unit dado un vector
            #Basicamente calcula la distancia euclidiana entre cada
            #el vector de ponderacion de la neurona y la entrada, y devuelve el
            #indice de la neurona que da el menor valor.
            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(self._weightage_vects, tf.stack([self._vect_input for i in range(m*n)])), 2), 1)),                                  0)

            #This will extract the location of the BMU based on the BMU's
            #index
            slice_input = tf.pad(tf.reshape(bmu_index, [1]),
                                 np.array([[0, 1]]))
            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input,
                                          tf.constant(np.array([1, 2]),dtype=tf.int64)),
                                 [2])

            #Esto extraera la ubicacion del BMU basado en el indice del BMU.
            learning_rate_op = tf.subtract(1.0, tf.div(self._iter_input,
                                                  self._n_iterations))
            _alpha_op = tf.multiply(alpha, learning_rate_op)
            _sigma_op = tf.multiply(sigma, learning_rate_op)

            #Para calcular los valores alfa y sigma basados en el numero de iteracion
            #Tasas para todas las neuronas, segun el numero de iteracion y la ubicacion
            #de la BMU
            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(
                self._location_vects, tf.stack(
                    [bmu_loc for i in range(m*n)])), 2), 1)
            neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(
                bmu_distance_squares, "float32"), tf.pow(_sigma_op, 2))))
            learning_rate_op = tf.multiply(_alpha_op, neighbourhood_func)

            #Finalmente, la op que usara learning_rate_op para actualizar
            #Los pesos de los vectores de todas las neuronas basados en un determinado
            #entrada
            learning_rate_multiplier = tf.stack([tf.tile(tf.slice(
                learning_rate_op, np.array([i]), np.array([1])), [dim])
                                               for i in range(m*n)])
            weightage_delta = tf.multiply(
                learning_rate_multiplier,
                tf.subtract(tf.stack([self._vect_input for i in range(m*n)]),
                       self._weightage_vects))
            new_weightages_op = tf.add(self._weightage_vects,weightage_delta)
            self._training_op = tf.assign(self._weightage_vects,new_weightages_op)


            ##INICIALIZAR SESIÓN
            self._sess = tf.Session()

            ##INICIALIZAR LAS VARIABLES
            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)

    def _neuron_locations(self, m, n):
        """
	Proporciona una por una las ubicaciones 2-D
	de las neuronas individuales en el SOM.
        """
        #Nested iterations over both dimensions
        #to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

    def train(self, input_vects):
        """
        Entrena el SOM.
	'input_vects' debe ser iterable en las matrices NumPy 1-D 
	con la dimensionalidad provista durante la inicializacion del SOM.
	Los vectores de pesos actuales para todas las neuronas (inicialmente
	aleatorios) se toman como condiciones de inicio para el entrenamiento.
        """
        #fig2 = plt.figure()

	# Iteraciones de entrenamiento
        for iter_no in tqdm(range(self._n_iterations)):
            # Entrenar con cada vector uno por uno
            for input_vect in input_vects:
                self._sess.run(self._training_op,
                               feed_dict={self._vect_input: input_vect,self._iter_input: iter_no})

        # Almacena una grid de centroides para recuperarla mas adelante
        centroid_grid = [[] for i in range(self._m)]
        self._weightages = list(self._sess.run(self._weightage_vects))
        self._locations = list(self._sess.run(self._location_vects))
        for i, loc in enumerate(self._locations):
            centroid_grid[loc[0]].append(self._weightages[i])
        #im_ani = animation.ArtistAnimation(fig2, centroid_grid, interval=50, repeat_delay=3000, blit=True)
        self._centroid_grid = centroid_grid
        #print(centroid_grid)


        self._trained = True
        #plt.show()

    def get_centroids(self):
        """
	Devuelve una lista de listas 'm', cada una de las cuales contiene
	las 'n' ubicaciones de centroides correspondientes como arrays NumPy 1-D.
        """
        if not self._trained:
            raise ValueError("SOM SOM aun no entrenado")
        return self._centroid_grid

    def map_vects(self, input_vects,other):
        """
	Asigna cada vector de entrada a la neurona relevante en la grid SOM.
	'input_vects' debe ser una iterable de las matrices NumPy 1-D con la
	dimensionalidad provista durante la inicializacion de este SOM.
	Devuelve una lista de matrices NumPy 1-D que contienen informacion
	(fila, columna) para cada vector de entrada (en el mismo orden),
	correspondiente a la neurona asignada.
        """

        if not self._trained:
            raise ValueError("SOM aun no entrenado")
        to_return=np.zeros((0,4))
        input_vects=np.array(input_vects)
        for vect,i in zip(input_vects,range(0,input_vects.shape[0])):
            to_return=np.insert(to_return,i,np.array((vect[0],vect[1],self._locations[min([i for i in range(len(self._weightages))],key=lambda x: np.linalg.norm(vect-self._weightages[x]))][0] , self._locations[min([i for i in range(len(self._weightages))],key=lambda x: np.linalg.norm(vect-self._weightages[x]))][1] )),0)
        return to_return
