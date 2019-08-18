/**********************************************************************
Copyright ©2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#define InputN 9		// number of neurons in the input layer
#define HN 40			// number of neurons in the hidden layer
#define OutN 1			// number of neurons in the output layer
#define datanum 80		// number of training samples

double sigmoid(double x) {
	return(1.0 / (1.0 + exp(-x)));
}
__kernel void helloworld(__global char* in, __global char* out,__global double* inputData,__global double* outputData,__global double* wingreso,__global double* vingreso,__global double* w,__global double* v, __global double* errors)
{
	int num = get_global_id(0);
	//out[num] = in[num] + 1;
	//w[num]=1;

	double hn_out[HN];
	double y_out[OutN];         // output layer
	double x=0;
	double y[OutN];	
	double x_out[InputN];
	double wi[InputN][HN];		// weights from input layer to hidden layer
	double vi[HN][OutN];			// weights from hidden layer to output layer

	double deltaw[InputN][HN];
	double deltav[HN][OutN];

	double hn_delta[HN];		// delta of hidden layer
	double y_delta[OutN];		// delta of output layer
	double error;
	double errlimit = 0.01;
	double alpha = 0.1, beta = 0.1;
	int loop = 0;
	int times = 500;
	int i, j, m;
	double max, min;
	double sumtemp;
	double errtemp;



	// Initializition
	for (i = 0; i<InputN; i++) {
		for (j = 0; j<HN; j++) {
			wi[i][j] = wingreso[j+HN*i];
			deltaw[i][j] = 0;
		}
	}
	for (i = 0; i<HN; i++) {
		for (j = 0; j<OutN; j++) {
			vi[i][j] = vingreso[j+OutN*i];
			deltav[i][j] = 0;
		}
	}

	// Training

	//while (loop < times || error >= errlimit) {
		//loop++;
		for(loop=0;loop<90;loop++){
		error = 0.0;
			
			
		for (int m = 0; m < datanum; m++) {

			// Feedforward
			max = 1;
			min = -1;
			for (i = 0; i<InputN; i++) {
				x_out[i] = inputData[m*InputN+i];
			}
			for (i = 0; i<InputN; i++) {
				x_out[i] = (x_out[i] - min) / (max - min);
			}
			for (i = 0; i<OutN; i++) {
				y[i] = outputData[m*OutN+i];
			}
			
			for (i = 0; i < HN; i++) {
				sumtemp = 0.0;
				for (j = 0; j < InputN; j++) {
					sumtemp += wi[j][i] * x_out[j];
				}
				//hn_out[i] = sigmoid(sumtemp);//
				hn_out[i] = (1.0 / (1.0 + exp(-sumtemp)));
			}
			for (i = 0; i < OutN; i++) {
				sumtemp = 0.0;
				for (j = 0; j < HN; j++)
					sumtemp += vi[j][i] * hn_out[j];
				x = sigmoid(sumtemp);
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			// Backpropagation
				
			for (i = 0; i<OutN; i++) {
				errtemp = y[i] -x;
				y_delta[i] = -errtemp * sigmoid(x) * (1.0 - sigmoid(x));
				error += errtemp * errtemp;
			}
				
			for (i = 0; i<HN; i++) {
				errtemp = 0.0;
				for (j = 0; j<OutN; j++)
					errtemp += y_delta[j] * vi[i][j];
				hn_delta[i] = errtemp * (1.0 + hn_out[i]) * (1.0 - hn_out[i]);
			}

			// Stochastic gradient descent
				
			for (i = 0; i<OutN; i++) {
				for (j = 0; j<HN; j++) {
					deltav[j][i] = alpha * deltav[j][i] + beta*y_delta[i]* hn_out[j];//;
					vi[j][i] -= deltav[j][i];
				}
			}

			for (i = 0; i<HN; i++) {
				for (j = 0; j<InputN; j++) {
					deltaw[j][i] = alpha * deltaw[j][i] + beta  * x_out[j]*hn_delta[i];
					wi[j][i] -= deltaw[j][i];
				}
			}
		}
		// Global error 
		error = error / 2;
		//if (loop % 1000 == 0) {
			//printf("Global Error = %f\n", error);
		//}
		//if (error < errlimit)
			//break;
			barrier(CLK_LOCAL_MEM_FENCE);
	}
	//
			for (i = 0; i<HN; i++) {
				for (j = 0; j<InputN; j++) {
					w[i*InputN+j]=wi[j][i];
				}
			}
			for (i = 0; i<HN; i++) {
				for (j = 0; j<OutN; j++) {
					v[i*OutN+j]=vi[i][j];
				}
			}
			//w[HN*InputN+1]=vi[0][0];
			//v[num]=2;
	errors[num]=error;//sigmoid(1);//1+2+2^2;//sin(0);
	
}
