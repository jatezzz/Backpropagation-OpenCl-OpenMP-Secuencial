/************************************************
* Backpropagation algorithm.
*
* Training a Neural Network, or an Autoencoder.
*
* Likewise, you can increase the number of layers
* to implement a deeper structure, to follow the
* trend of "Deep Learning". But bear in mind that
* DL has lots of beautiful tricks, and merely
* making it deeper will not yield good results!!
*
* Designed by Junbo Zhao, 12/14/2013
************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define InputN 9		// number of neurons in the input layer
#define HN 40			// number of neurons in the hidden layer
#define OutN 1			// number of neurons in the output layer
#define datanum 91		// number of training samples

void main() {
	double	starttime,
		stoptime,
		etime,
		wtick;

	double sigmoid(double);
	char result[16] = "";
	char buffer[200];
	double x_out[InputN];		// input layer
	double hn_out[HN];			// hidden layer
	double y_out[OutN];         // output layer
	double y[OutN];				// expected output layer
	double w[InputN][HN];		// weights from input layer to hidden layer
	double v[HN][OutN];			// weights from hidden layer to output layer

	double deltaw[InputN][HN];
	double deltav[HN][OutN];

	double hn_delta[HN];		// delta of hidden layer
	double y_delta[OutN];		// delta of output layer
	double error;
	double errlimit = 0.001;
	double alpha = 0.1, beta = 0.1;
	int loop = 0;
	int times = 50000;
	int i, j, m;
	double max, min;
	double sumtemp;
	double errtemp;

	//Apertura de archivo
	srand(time(NULL));

#pragma warning(disable:4996)
	FILE *fp;
	if ((fp = fopen("test2.txt", "r")) == NULL) {
		printf("Cannot open file.\n");
		exit(1);
	}

	i = 0;
	int entradas[9][208], salidas[208];
	while (fscanf(fp, "%d %d %d %d %d %d %d %d %d %d", &entradas[0][i], &entradas[1][i], &entradas[2][i], &entradas[3][i], &entradas[4][i], &entradas[5][i], &entradas[6][i], &entradas[7][i], &entradas[8][i], &salidas[i]) != EOF) {
		printf("%d %d %d %d %d %d %d %d %d %d", entradas[0][i], entradas[1][i], entradas[2][i], entradas[3][i], entradas[4][i], entradas[5][i], entradas[6][i], entradas[7][i], entradas[8][i], salidas[i]);
		printf("\t%i\n", i);
		i++;
	}


	// training set
	struct {
		double input[InputN];
		double teach[OutN];
	}data[datanum];

	// Generate data samples
	for (m = 0; m<datanum; m++) {
		for (i = 0; i < InputN; i++) {
			data[m].input[i] = (double)entradas[i][m];
		}
		for (i = 0; i < OutN; i++) {
			data[m].teach[i] = (double)salidas[m];
		}
	}


	// Initializition
	for (i = 0; i<InputN; i++) {
		for (j = 0; j<HN; j++) {
			w[i][j] = ((double)rand() / 32767.0) * 2 - 1;
			deltaw[i][j] = 0;
		}
	}
	for (i = 0; i<HN; i++) {
		for (j = 0; j<OutN; j++) {
			v[i][j] = ((double)rand() / 32767.0) * 2 - 1;
			deltav[i][j] = 0;
		}
	}
	// Training
	starttime = omp_get_wtime();
	while (loop < times) {
		loop++;
		error = 0.0;

		for (m = 0; m<datanum; m++) {
			// Feedforward

			max = 1;
			min = -1;
			for (i = 0; i<InputN; i++) {
				x_out[i] = data[m].input[i];

			}
			if ((max - min)>0) {
			for (i = 0; i < InputN; i++) {
				x_out[i] = (x_out[i] - min) / (max - min);
			}
			}

			for (i = 0; i<OutN; i++) {
				y[i] = data[m].teach[i] / 9;
			}
			for (i = 0; i<HN; i++) {
				sumtemp = 0.0;
				for (j = 0; j < InputN; j++) {
					sumtemp += w[j][i] * x_out[j];
				}
				hn_out[i] = sigmoid(sumtemp);		// sigmoid serves as the activation function
			}
			for (i = 0; i<OutN; i++) {
				sumtemp = 0.0;
				for (j = 0; j < HN; j++) {
					sumtemp += v[j][i] * hn_out[j];
				}
				y_out[i] = sigmoid(sumtemp);
			}
			// Backpropagation
			for (i = 0; i<OutN; i++) {
				errtemp = y[i] - y_out[i];

				y_delta[i] = -errtemp * sigmoid(y_out[i]) * (1.0 - sigmoid(y_out[i]));

				error += errtemp * errtemp;
			}
			
			// Stochastic gradient descent
			
					for (i = 0; i<OutN; i++) {
						for (j = 0; j<HN; j++) {
							deltav[j][i] = alpha * deltav[j][i] + beta * y_delta[i] * hn_out[j];
							v[j][i] -= deltav[j][i];
						}
					}
				

					for (i = 0; i<HN; i++) {
						errtemp = 0.0;
						for (j = 0; j<OutN; j++)
							errtemp += y_delta[j] * v[i][j];
						hn_delta[i] = errtemp * (1.0 + hn_out[i]) * (1.0 - hn_out[i]);
					}
					for (i = 0; i<HN; i++) {
						for (j = 0; j<InputN; j++) {
							deltaw[j][i] = alpha * deltaw[j][i] + beta * hn_delta[i] * x_out[j];
							w[j][i] -= deltaw[j][i];
						}
					}
				
			
	
			
		}

		// Global error 
		error = error / 2;
		if (loop % 1000 == 0) {
			//printf("Global Error = %f\n", error);
		}
		if (error < errlimit)
			break;

		printf("The %d th training, error: %f\n", loop, error);
	}
	stoptime = omp_get_wtime();

	//Impresión de resultados
	printf("\nError final: %f\n", error);
	etime = stoptime - starttime;
	wtick = omp_get_wtick();
	printf("tick = %f seconds \n", wtick);
	//printf("Elapsed time in for-loop = %f miliseconds \n", etime / wtick);
	printf("Elapsed time in for-loop = %f seconds \n", etime);

	//Test Feedforward
	max = 1;
	min = -1;
	float test_input[] = { 0,0,-1,0,1,0,0,0,0 };
	printf("\nEntrada: ");
	for (i = 0; i<InputN; i++) {
		x_out[i] = test_input[i];
		printf("%2.1f ", x_out[i]);
	}
	for (i = 0; i<InputN; i++) {
		x_out[i] = (x_out[i] - min) / (max - min);
	}

	for (int i = 0; i < HN; i++) {
		sumtemp = 0.0;
		for (j = 0; j < InputN; j++)
			sumtemp += w[j][i] * x_out[j];
		hn_out[i] = sigmoid(sumtemp);		// sigmoid serves as the activation function
	}
	for (i = 0; i<OutN; i++) {
		sumtemp = 0.0;
		for (j = 0; j<HN; j++)
			sumtemp += v[j][i] * hn_out[j];
		y_out[i] = sigmoid(sumtemp);
		printf("\nResultado a la entrada %i= %f\n", i, y_out[i] * 9);
	}
	//Impresión de tablero
	printf("\tTIC TAC TOE\n");
	printf("\t___|___|___\n");
	printf("\t___|_O_|___\n");
	printf("\t   |   |   \n");
	entradas[5 - 1][0] = 1;
	loop = 0;
	int pos = 0;
	char simbolos[] = ".........";
	while (loop < 6) {
		loop++;
		printf("Ingresa proxima posición[1-4]U[6-9] : ");
		do { scanf("%d", &pos); } while (pos == 5);
		entradas[pos-1][0] = -1;
		//Impresión
		for (i = 0; i < 9; i++) {
			if (entradas[i][0] == 0) {
				simbolos[i] = '.';
			}
			else if (entradas[i][0] == -1) {
				simbolos[i] = 'X';
			}
			else if (entradas[i][0] == 1) {
				simbolos[i] = 'O';
			}
		}
		printf("\t_%c_|_%c_|_%c_\n", simbolos[0], simbolos[1], simbolos[2]);
		printf("\t_%c_|_%c_|_%c_\n", simbolos[3], simbolos[4], simbolos[5]);
		printf("\t %c | %c | %c \n", simbolos[6], simbolos[7], simbolos[8]);

		//Test Feedforward
		max = 1;
		min = -1;
		for (i = 0; i<InputN; i++) {
			x_out[i] = entradas[i][0];
			if (max < x_out[i])
				max = x_out[i];
			if (min > x_out[i])
				min = x_out[i];
		}
		for (i = 0; i<InputN; i++) {
			x_out[i] = (x_out[i] - min) / (max - min);
		}

		for (int i = 0; i < HN; i++) {
			sumtemp = 0.0;
			for (j = 0; j < InputN; j++)
				sumtemp += w[j][i] * x_out[j];
			hn_out[i] = sigmoid(sumtemp);		// sigmoid serves as the activation function
		}
		for (i = 0; i<OutN; i++) {
			sumtemp = 0.0;
			for (j = 0; j<HN; j++)
				sumtemp += v[j][i] * hn_out[j];
			y_out[i] = sigmoid(sumtemp);
			printf("Salida Real %i= %f\n", i, y_out[i] * 9);
			printf("Salida Ajustada %i= %f\n", i, round(y_out[i] * 9));
		}
		int pos_m = round(y_out[0] * 9);
		
		entradas[pos_m - 1][0] = 1;
		printf("Respuesta: %d,%d\n", pos_m,entradas[pos_m - 1][0]);
	
		//Impresión
		for (i = 0; i < 9; i++) {
			if (entradas[i][0] == 0) {
				simbolos[i] = '.';
			}
			else if (entradas[i][0] == -1) {
				simbolos[i] = 'X';
			}
			else if (entradas[i][0] == 1) {
				simbolos[i] = 'O';
			}
		}
		printf("\t_%c_|_%c_|_%c_\n", simbolos[0], simbolos[1], simbolos[2]);
		printf("\t_%c_|_%c_|_%c_\n", simbolos[3], simbolos[4], simbolos[5]);
		printf("\t %c | %c | %c \n", simbolos[6], simbolos[7], simbolos[8]);
		if ((entradas[6][0] == 1 & entradas[4][0] == 1 & entradas[2][0] == 1)|| (entradas[8][0] == 1 & entradas[4][0] == 1 & entradas[0][0] == 1) || (entradas[1][0] == 1 & entradas[4][0] == 1 & entradas[7][0] == 1) || (entradas[3][0] == 1 & entradas[4][0] == 1 & entradas[5][0] == 1) || (entradas[0][0] == 1 & entradas[1][0] == 1 & entradas[2][0] == 1) || (entradas[0][0] == 1 & entradas[3][0] == 1 & entradas[6][0] == 1) || (entradas[2][0] == 1 & entradas[5][0] == 1 & entradas[8][0] == 1) || (entradas[6][0] == 1 & entradas[7][0] == 1 & entradas[8][0] == 1)) {
			printf("\t-------------------¡Gano!--------------------- \n");
			scanf("%d", &pos);
			if (pos == 0) {
				for (i = 0; i < 9; i++) {
					entradas[i][0] = 0;
				}
				//Impresión de tablero
				printf("\tTIC TAC TOE\n");
				printf("\t_._|_._|_._\n");
				printf("\t_._|_O_|_._\n");
				printf("\t . | . | . \n");
				entradas[5 - 1][0] = 1;
				loop = 0;
				int pos = 0;
			}
			else {
				break;
			}
			
		}else if(pos_m==0){
			printf("\t-------------------¡Nadie Gana!--------------------- \n");
			scanf("%d", &pos);
			if (pos == 0) {
				for (i = 0; i < 9; i++) {
					entradas[i][0] = 0;
				}
				//Impresión de tablero
				printf("\tTIC TAC TOE\n");
				printf("\t_._|_._|_._\n");
				printf("\t_._|_O_|_._\n");
				printf("\t . | . | . \n");
				entradas[5 - 1][0] = 1;
				loop = 0;
				int pos = 0;
			}
			else {
				break;
			}
		}
	}
}

// sigmoid serves as avtivation function
double sigmoid(double x) {
	return(1.0 / (1.0 + exp(-x)));
}