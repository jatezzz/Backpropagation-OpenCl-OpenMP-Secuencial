An�lisis de paralelismo en aplicaci�n Tic Tac Toe mediante Redes Neuronales
=================
>**Autores: ** Carchi Suquillo Mar�a Bel�n, Trujillo Estrella John Andr�s
Departamento de El�ctrica y Electr�nica
Universidad de las Fuerzas Armadas ESPE
Sangolqu�, Ecuador
**Emails:**mbcarchi@espe.edu.ec, jatrujillo6@espe.edu.ec

**Introduci�n**� Se presenta una comparaci�n entre dos tipos actuales de programaci�n paralela, los cuales son distinguidos por su generalidad y rendimiento cuando son comparadas las soluciones de cada una de estas con respecto a la otra, espec�ficamente OpenMP y OpenCL, los cuales se han utilizado para la aplicaci�n espec�fica Tic Tac Toe la cual resuelve esta problem�tica a trav�s de redes neuronales. 
Los principales resultados que se obtienen con estos modelos han sido.

**INTRODUCCI�N**
En la actualidad  tanto para computadores de uso com�n como para computadoras de alto rendimiento. Estos sistemas utilizan multithreading de modo que los hilos que esperan datos puedan cambiar de funci�n y ejecutar diferentes instrucciones, mejorando de esta forma su eficiencia. Para lograr un mejor desempe�o en estos sistemas se requiere una estructuraci�n expl�cita de aplicaciones para aprovechar al m�ximo el paralelismo. La tecnolog�a multicore ofrece un rendimiento eficiente y OpenMP ha sido dise�ada como un modelo de programaci�n para aprovecharla. [1]

Los computadores modernos basan su alto rendimiento en incluir tantas t�cnicas de paralelizaci�n como sea posible. Este enfoque aumenta enormemente el rendimiento te�rico. Sin embargo, dicho rendimiento se consigue a costa de cargar ese paralelismo en el software, por lo que la responsabilidad de conseguir un buen rendimiento recae sobre el software debido a que los programas deben tener la capacidad de adaptarse para aprovechar todas las ventajas de las caracter�sticas paralelas que provee el software. [2]

Existen ciertos factores que generan mayor impacto al analizar el rendimiento, entre estos sobresalen el acceso a la memoria, el aprovechamiento de la vectorizaci�n, la minimizaci�n de trabajo innecesario y el overhead para el c�lculo paralelo, existen modelos de programaci�n que proporcionan suficiente control sobre estos factores cr�ticos, como lo son OpenCL, que motiva al usuario a escribir c�digo que puede ser vectorizado por el compilador OpenCL, en cambio OpenMP se basa en el uso expl�cito de #pragma para habilitar la vectorizaci�n. [1], [3]

OpenMp es uno de los principales lenguajes utilizados en entornos de memoria compartida, el que proporciona un API que, mediante directivas de compilador y llamadas a subrutinas, proporciona paralelismo de datos. La unidad base es el hilo, y cada uno tiene acceso a variables en cach� compartida o RAM. Los mejores resultados se observan cuando el acceso a datos compartidos tiene bajo coste. [2]

OpenCL es la alternativa libre a tecnolog�as como CUDA de NVidia, que intenta aprovechar la potencia de los procesadores gr�ficos para realizar operaciones intensas repartidas entre el procesador del equipo y la GPU de cualquier tarjeta gr�fica compatible, al adaptar las aplicaciones y aprovechamiento de las ventajas de OpenCL no dependen del hardware o sistema operativo que tenga la m�quina. [4]

Las redes neuronales emulan ciertas caracter�sticas propias de los humanos, como la capacidad de memorizar y de asociar hechos, en definitiva, es un modelo artificial y simplificado del cerebro humano, siento un sistema para el tratamiento de la informaci�n que consiste de unidades de procesamiento que intercambian datos e informaci�n, utilizadas para el reconocimiento de patrones y con capacidad de aprendizaje y mejora de respuesta. [5]

El m�todo m�s utilizado para la ense�anza de una red neuronal es el de retropropagaci�n, que consiste en la autoadaptaci�n de los pesos de las neuronas de las capas intermedias para aprender la relaci�n que existe entre un conjunto de patrones de entrada y sus salidas correspondientes, como resultado la red debe encontrar una representaci�n interna que le permita generar las salidas deseadas cuando se le proporcionan entradas de entrenamiento y adem�s a entradas no presentadas durante la etapa de aprendizaje. [6]

	Es por esto que un modelo de red neuronal dirigido a aplicaci�n tiene una arquitectura directamente ligada a la necesidad espec�fica, que en este caso es Tic Tac Toe, permitiendo debido a su estructura realizar una comparaci�n entre las tecnolog�as OpenMP y OpenCL. 
	De acuerdo a estos criterios el resto del art�culo se organiza de la siguiente manera. En la secci�n II se detalla el problema a resolver con el desarrollo del mismo, la secci�n III est� centrada en la implementaci�n que se realiza como soluci�n al problema planteado previamente, la secci�n IV corresponde al an�lisis de los resultados obtenidos y las conclusiones y recomendaciones que se han obtenido se describen en las secciones V y VI respectivamente.

**DESCRIPCI�N DEL PROBLEMA**

Para el an�lisis de factores de performance computacional con diferentes modelos de programaci�n se describe una aplicaci�n para solucionar el juego Tic Tac Toe, el cual mediante una red neuronal con algoritmo de retropropagaci�n obtiene la mejor respuesta a una jugada previa, este programa cuenta con secciones paralelizables que ser�n desarrolladas mediante OpenMP y OpenCL y se analiza su perfomance.  
A.	Tic Tac Toe mediante redes neuronales
Como se observa en la Figura 1. El esquema de la red neuronal se basa en la interacci�n de cada una de las capas que lo componen, las cuales sometidas a factores de entrenamiento y mediante esto la asignaci�n de pesos se pueden obtener los resultados deseados.
La red neuronal implementada consta de 9 neuronas de entrada correspondientes a las 9 posibles posiciones del tablero, 40 neuronas en la capa escondida y una neurona en la capa de salida correspondiente a la respuesta a un movimiento.

B.	Algoritmo de retropropagaci�n 

Una red con retropropagaci�n tiene dos fases, la primera se la da hacia adelante en la cual el patr�n de entrada es presentado a la red y se propaga a trav�s de las capas hasta llegar a la de salida, una vez que se obtienen estos valores de salida de la red, se inicia la segunda fase en la que se comparan estos valores con la salida esperada para as� obtener el error, ajustando de esta forma el peso de la �ltima capa y pas�ndolo a la capa anterior con una retropropagaci�n del error y continuando el proceso hasta llegar a la primera capa. De esta manera se modifican los pesos de las conexiones de la red para cada uno de los patrones de aprendizaje.

C.	Retropropagaci�n mediante OpenMP

Devido a los m�ltiples procesos que involucra el algortimo, una forma de acelerarlo es utilizando los diferentes cores del CPu, con ello tras distribuir el trabajo en muchas de sus secciones, el mismo podr�a ser ejecutado de manera m�s r�pida.
D.	Retropropagaci�n mediante OpenCL
Al igual que una paralelizaci�n por CPU, se conoce que muchas computadoras cuentan de igual forma con GPUs enfocadas en procesamiento de im�genes, por ello, se har� uso de las mismas para que sean estas las que ejecuten el c�digo en base a las entradas y salidas de la red en fase de entrenamiento.
**IMPLEMENTACI�N**

Se realiza la implementaci�n del juego tic tac toe, con la caracter�stica de que los movimientos que realiza el computador han sido realizados a partir del entrenamiento de una red neuronal.	
A.	Descripci�n de la red neuronal

La red neuronal se conforma por 9 entradas y 1 salida, con una capa oculta entre estas, cada una de estas entradas representa una casilla del tablero, las entradas corresponden al estado del tablero y la salida al movimiento que la red neuronal ha calculado como mejor opci�n de respuesta. 

B.	Entrenamiento de la red neuronal

El entrenamiento se lo realiza en la ejecuci�n inicial del programa, es decir antes de que empiece el juego y se debe esperar a que esta termine su entrenamiento, en esta parte son analizados una serie de tableros en distintos estados como entradas y como salidas se pasan estos mismos tableros con el movimiento que se considera deber�a hacerse. 

C.	Salida de la red neuronal

A partir de las entradas que se dan la red neuronal procesa mediante una comparaci�n de pesos que resultan de una valoraci�n del error que se le asigna, en el cual en un inicio se asignan pesos aleatorios y se eval�a la respuesta y a partir de este se retropropagan estos valores para obtener una mejor respuesta con un menor error.

D.	Programaci�n con OpenMP

En la figura 3 se muestra una secci�n del c�digo implementado, en donde se implementa una paralelizaci�n de un loop bastante extenso puesto que se encarga de cada una de las capas ocultas y a su vez en cada una de las entradas, permitiendo con ello una paralelizaci�n y distribuci�n del c�mputo.
E.	Programaci�n con OpenCL
De igual forma se sigue el procedimiento de implementaci�n de un programa en paralelo con OpenCl, debido al cual se procede a implantar el mismo c�digo dentro del kernel a ser ejecutado por la funci�n principal.

**RESULTADOS**

Los resultados se analizan de acuerdo al error que se generan con respecto a las �pocas que han transcurrido, la figura muestra la reducci�n del error en cuanto incrementan las �pocas de entrenamiento, con ello se muestra una familiaridad con una gr�fica exponencial, por tal motivo se demuestra que para el ejemplo si bien se necesitaron cerca de 8000 iteraciones, ya desde la �poca 950, se observa un error inferior a 0.4 veces con el valor real.
En la tabla a continuaci�n se ejemplifica parte de los datos representados para el entrenamiento del juego Tic Tac Toe, en  donde se observa la ejemplificaci�n del oponente como una X con un valor de -1, y la m�quina como O, con un valor de 1. Se observa a su vez que dicha aplicaci�n parte del hecho que la m�quina empieza el juego con una marca O en el centro del tablero, esto para poder disminuir el n�mero de combinaciones.


Tabla 1 Patrones de entrenamiento
Entradas	Respuesta	Representaci�n Textual
1	2	3	4	5	6	7	8	9		
.	.	.	.	.	.	.	.	.	5	0 0 0 0 0 0 0 0 0 5
x	.	.	.	o	.	.	.	.	7	-1 0 0 0 1 0 0 0 0 7
.	x	.	.	o	.	.	.	.	7	0 -1 0 0 1 0 0 0 0 7
.	.	x	.	o	.	.	.	.	1	0 0 -1 0 1 0 0 0 0 1
.	.	.	x	o	.	.	.	.	9	0 0 0 -1 1 0 0 0 0 9
.	.	.	.	o	x	.	.	.	1	0 0 0 0 1 -1 0 0 0 1
.	.	.	.	o	.	x	.	.	9	0 0 0 0 1 0 -1 0 0 9
.	.	.	.	o	.	.	x	.	3	0 0 0 0 1 0 0 -1 0 3
.	.	.	.	o	.	.	.	x	3	0 0 0 0 1 0 0 0 -1 3
x	x	.	.	o	.	o	.	.	3	-1 -1 0 0 1 0 1 0 0 3
x	.	.	x	o	.	o	.	.	3	-1 0 0 -1 1 0 1 0 0 3
x	.	.	.	o	x	o	.	.	3	-1 0 0 0 1 -1 1 0 0 3
x	.	.	.	o	.	o	x	.	3	-1 0 0 0 1 0 1 -1 0 3
x	.	.	.	o	.	o	.	x	3	-1 0 0 0 1 0 1 0 -1 3


Se muestra entonces el resultado obtenido con un error final de 0.02 veces el valor real, y con ello, se observa la respuesta a un valor de ejemplo y un valor dado por el usuario, lo que demuestra la exactitud del valor y el valor bajo de error, valido para la aplicaci�n deseada.
Tras los experimentos realizados para un n�mero de iteraciones de 90, se observa que el tiempo de ejecuci�n se ve claramente diferenciado debido a que, al ser un n�mero bajo de iteraciones, el c�digo adicional en cada paralelizaci�n interviene en gran medida provocando un speedup menor a 1 para la utilizaci�n de OpenmMP y OpenCL. 

**CONCLUSIONES**

�	En el desarrollo del presente documento se realiza una comparaci�n de dos diferentes tipos de programaci�n paralela con respecto a OpenMP y OpenCL, en los cuales mediante una evaluaci�n se ha determinado que

�	Las evaluaciones de estas tecnolog�as han demostrado varios resultados, como la mejora significativa en el rendimiento obtenido por las aplicaciones 
�	EL algoritmo de retropropagaci�n cuenta con m�ltiples segmentos, y son estos los que deben ser estudiados para poder analizar las formas de paralelizar cada una de las mismas y con ello encontrar la mejor combinaci�n con el menor tiempo de ejecuci�n en tareas como la fase completa de entrenamiento.

**RECOMENDACIONES**

�	En la etapa de implementaci�n es necesario conocer el conjunto de instrucciones disponibles de cada uno de los modelos de programaci�n, y sobre todo su similitud para identificar correctamente las partes en las que se van a trabajar y las funciones equivalentes en cada uno de estos.

�	Si bien la implementaci�n de OpenMP es sencilla, en comparaci�n con otro tipo de modelos de programaci�n, esta debe generarse con criterio puesto que no siempre reducir� el tiempo de ejecuci�n sobre todo en aplicaciones con dependencia de datos.