Análisis de paralelismo en aplicación Tic Tac Toe mediante Redes Neuronales
=================
>**Autores: ** Carchi Suquillo María Belén, Trujillo Estrella John Andrés
Departamento de Eléctrica y Electrónica
Universidad de las Fuerzas Armadas ESPE
Sangolquí, Ecuador
**Emails:**mbcarchi@espe.edu.ec, jatrujillo6@espe.edu.ec

**Introdución**— Se presenta una comparación entre dos tipos actuales de programación paralela, los cuales son distinguidos por su generalidad y rendimiento cuando son comparadas las soluciones de cada una de estas con respecto a la otra, específicamente OpenMP y OpenCL, los cuales se han utilizado para la aplicación específica Tic Tac Toe la cual resuelve esta problemática a través de redes neuronales. 
Los principales resultados que se obtienen con estos modelos han sido.

**INTRODUCCIÓN**
En la actualidad  tanto para computadores de uso común como para computadoras de alto rendimiento. Estos sistemas utilizan multithreading de modo que los hilos que esperan datos puedan cambiar de función y ejecutar diferentes instrucciones, mejorando de esta forma su eficiencia. Para lograr un mejor desempeño en estos sistemas se requiere una estructuración explícita de aplicaciones para aprovechar al máximo el paralelismo. La tecnología multicore ofrece un rendimiento eficiente y OpenMP ha sido diseñada como un modelo de programación para aprovecharla. [1]

Los computadores modernos basan su alto rendimiento en incluir tantas técnicas de paralelización como sea posible. Este enfoque aumenta enormemente el rendimiento teórico. Sin embargo, dicho rendimiento se consigue a costa de cargar ese paralelismo en el software, por lo que la responsabilidad de conseguir un buen rendimiento recae sobre el software debido a que los programas deben tener la capacidad de adaptarse para aprovechar todas las ventajas de las características paralelas que provee el software. [2]

Existen ciertos factores que generan mayor impacto al analizar el rendimiento, entre estos sobresalen el acceso a la memoria, el aprovechamiento de la vectorización, la minimización de trabajo innecesario y el overhead para el cálculo paralelo, existen modelos de programación que proporcionan suficiente control sobre estos factores críticos, como lo son OpenCL, que motiva al usuario a escribir código que puede ser vectorizado por el compilador OpenCL, en cambio OpenMP se basa en el uso explícito de #pragma para habilitar la vectorización. [1], [3]

OpenMp es uno de los principales lenguajes utilizados en entornos de memoria compartida, el que proporciona un API que, mediante directivas de compilador y llamadas a subrutinas, proporciona paralelismo de datos. La unidad base es el hilo, y cada uno tiene acceso a variables en caché compartida o RAM. Los mejores resultados se observan cuando el acceso a datos compartidos tiene bajo coste. [2]

OpenCL es la alternativa libre a tecnologías como CUDA de NVidia, que intenta aprovechar la potencia de los procesadores gráficos para realizar operaciones intensas repartidas entre el procesador del equipo y la GPU de cualquier tarjeta gráfica compatible, al adaptar las aplicaciones y aprovechamiento de las ventajas de OpenCL no dependen del hardware o sistema operativo que tenga la máquina. [4]

Las redes neuronales emulan ciertas características propias de los humanos, como la capacidad de memorizar y de asociar hechos, en definitiva, es un modelo artificial y simplificado del cerebro humano, siento un sistema para el tratamiento de la información que consiste de unidades de procesamiento que intercambian datos e información, utilizadas para el reconocimiento de patrones y con capacidad de aprendizaje y mejora de respuesta. [5]

El método más utilizado para la enseñanza de una red neuronal es el de retropropagación, que consiste en la autoadaptación de los pesos de las neuronas de las capas intermedias para aprender la relación que existe entre un conjunto de patrones de entrada y sus salidas correspondientes, como resultado la red debe encontrar una representación interna que le permita generar las salidas deseadas cuando se le proporcionan entradas de entrenamiento y además a entradas no presentadas durante la etapa de aprendizaje. [6]

	Es por esto que un modelo de red neuronal dirigido a aplicación tiene una arquitectura directamente ligada a la necesidad específica, que en este caso es Tic Tac Toe, permitiendo debido a su estructura realizar una comparación entre las tecnologías OpenMP y OpenCL. 
	De acuerdo a estos criterios el resto del artículo se organiza de la siguiente manera. En la sección II se detalla el problema a resolver con el desarrollo del mismo, la sección III está centrada en la implementación que se realiza como solución al problema planteado previamente, la sección IV corresponde al análisis de los resultados obtenidos y las conclusiones y recomendaciones que se han obtenido se describen en las secciones V y VI respectivamente.

**DESCRIPCIÓN DEL PROBLEMA**

Para el análisis de factores de performance computacional con diferentes modelos de programación se describe una aplicación para solucionar el juego Tic Tac Toe, el cual mediante una red neuronal con algoritmo de retropropagación obtiene la mejor respuesta a una jugada previa, este programa cuenta con secciones paralelizables que serán desarrolladas mediante OpenMP y OpenCL y se analiza su perfomance.  
A.	Tic Tac Toe mediante redes neuronales
Como se observa en la Figura 1. El esquema de la red neuronal se basa en la interacción de cada una de las capas que lo componen, las cuales sometidas a factores de entrenamiento y mediante esto la asignación de pesos se pueden obtener los resultados deseados.
La red neuronal implementada consta de 9 neuronas de entrada correspondientes a las 9 posibles posiciones del tablero, 40 neuronas en la capa escondida y una neurona en la capa de salida correspondiente a la respuesta a un movimiento.

B.	Algoritmo de retropropagación 

Una red con retropropagación tiene dos fases, la primera se la da hacia adelante en la cual el patrón de entrada es presentado a la red y se propaga a través de las capas hasta llegar a la de salida, una vez que se obtienen estos valores de salida de la red, se inicia la segunda fase en la que se comparan estos valores con la salida esperada para así obtener el error, ajustando de esta forma el peso de la última capa y pasándolo a la capa anterior con una retropropagación del error y continuando el proceso hasta llegar a la primera capa. De esta manera se modifican los pesos de las conexiones de la red para cada uno de los patrones de aprendizaje.

C.	Retropropagación mediante OpenMP

Devido a los múltiples procesos que involucra el algortimo, una forma de acelerarlo es utilizando los diferentes cores del CPu, con ello tras distribuir el trabajo en muchas de sus secciones, el mismo podría ser ejecutado de manera más rápida.
D.	Retropropagación mediante OpenCL
Al igual que una paralelización por CPU, se conoce que muchas computadoras cuentan de igual forma con GPUs enfocadas en procesamiento de imágenes, por ello, se hará uso de las mismas para que sean estas las que ejecuten el código en base a las entradas y salidas de la red en fase de entrenamiento.
**IMPLEMENTACIÓN**

Se realiza la implementación del juego tic tac toe, con la característica de que los movimientos que realiza el computador han sido realizados a partir del entrenamiento de una red neuronal.	
A.	Descripción de la red neuronal

La red neuronal se conforma por 9 entradas y 1 salida, con una capa oculta entre estas, cada una de estas entradas representa una casilla del tablero, las entradas corresponden al estado del tablero y la salida al movimiento que la red neuronal ha calculado como mejor opción de respuesta. 

B.	Entrenamiento de la red neuronal

El entrenamiento se lo realiza en la ejecución inicial del programa, es decir antes de que empiece el juego y se debe esperar a que esta termine su entrenamiento, en esta parte son analizados una serie de tableros en distintos estados como entradas y como salidas se pasan estos mismos tableros con el movimiento que se considera debería hacerse. 

C.	Salida de la red neuronal

A partir de las entradas que se dan la red neuronal procesa mediante una comparación de pesos que resultan de una valoración del error que se le asigna, en el cual en un inicio se asignan pesos aleatorios y se evalúa la respuesta y a partir de este se retropropagan estos valores para obtener una mejor respuesta con un menor error.

D.	Programación con OpenMP

En la figura 3 se muestra una sección del código implementado, en donde se implementa una paralelización de un loop bastante extenso puesto que se encarga de cada una de las capas ocultas y a su vez en cada una de las entradas, permitiendo con ello una paralelización y distribución del cómputo.
E.	Programación con OpenCL
De igual forma se sigue el procedimiento de implementación de un programa en paralelo con OpenCl, debido al cual se procede a implantar el mismo código dentro del kernel a ser ejecutado por la función principal.

**RESULTADOS**

Los resultados se analizan de acuerdo al error que se generan con respecto a las épocas que han transcurrido, la figura muestra la reducción del error en cuanto incrementan las épocas de entrenamiento, con ello se muestra una familiaridad con una gráfica exponencial, por tal motivo se demuestra que para el ejemplo si bien se necesitaron cerca de 8000 iteraciones, ya desde la época 950, se observa un error inferior a 0.4 veces con el valor real.
En la tabla a continuación se ejemplifica parte de los datos representados para el entrenamiento del juego Tic Tac Toe, en  donde se observa la ejemplificación del oponente como una X con un valor de -1, y la máquina como O, con un valor de 1. Se observa a su vez que dicha aplicación parte del hecho que la máquina empieza el juego con una marca O en el centro del tablero, esto para poder disminuir el número de combinaciones.


Tabla 1 Patrones de entrenamiento
Entradas	Respuesta	Representación Textual
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


Se muestra entonces el resultado obtenido con un error final de 0.02 veces el valor real, y con ello, se observa la respuesta a un valor de ejemplo y un valor dado por el usuario, lo que demuestra la exactitud del valor y el valor bajo de error, valido para la aplicación deseada.
Tras los experimentos realizados para un número de iteraciones de 90, se observa que el tiempo de ejecución se ve claramente diferenciado debido a que, al ser un número bajo de iteraciones, el código adicional en cada paralelización interviene en gran medida provocando un speedup menor a 1 para la utilización de OpenmMP y OpenCL. 

**CONCLUSIONES**

•	En el desarrollo del presente documento se realiza una comparación de dos diferentes tipos de programación paralela con respecto a OpenMP y OpenCL, en los cuales mediante una evaluación se ha determinado que

•	Las evaluaciones de estas tecnologías han demostrado varios resultados, como la mejora significativa en el rendimiento obtenido por las aplicaciones 
•	EL algoritmo de retropropagación cuenta con múltiples segmentos, y son estos los que deben ser estudiados para poder analizar las formas de paralelizar cada una de las mismas y con ello encontrar la mejor combinación con el menor tiempo de ejecución en tareas como la fase completa de entrenamiento.

**RECOMENDACIONES**

•	En la etapa de implementación es necesario conocer el conjunto de instrucciones disponibles de cada uno de los modelos de programación, y sobre todo su similitud para identificar correctamente las partes en las que se van a trabajar y las funciones equivalentes en cada uno de estos.

•	Si bien la implementación de OpenMP es sencilla, en comparación con otro tipo de modelos de programación, esta debe generarse con criterio puesto que no siempre reducirá el tiempo de ejecución sobre todo en aplicaciones con dependencia de datos.