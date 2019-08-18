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

// For clarity,error checking has been omitted.

#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <omp.h>

#define SUCCESS 0
#define FAILURE 1

#define InputN 9		// number of neurons in the input layer
#define HN 40			// number of neurons in the hidden layer
#define OutN 1			// number of neurons in the output layer
#define datanum 80		// number of training samples


using namespace std;

/* convert the kernel file into a string */
int convertToString(const char *filename, std::string& s)
{
	size_t size;
	char*  str;
	std::fstream f(filename, (std::fstream::in | std::fstream::binary));

	if(f.is_open())
	{
		size_t fileSize;
		f.seekg(0, std::fstream::end);
		size = fileSize = (size_t)f.tellg();
		f.seekg(0, std::fstream::beg);
		str = new char[size+1];
		if(!str)
		{
			f.close();
			return 0;
		}

f.read(str, fileSize);
		f.close();
		str[size] = '\0';
		s = str;
		delete[] str;
		return 0;
	}
	cout<<"Error: failed to open file\n:"<<filename<<endl;
	return FAILURE;
}

int main(int argc, char* argv[])
{
	double	starttime,
		stoptime,
		etime,
		wtick;

	/*Step1: Getting platforms and choose an available one.*/
	cl_uint numPlatforms;	//the NO. of platforms
	cl_platform_id platform = NULL;	//the chosen platform
	cl_int	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (status != CL_SUCCESS)
	{
		cout << "Error: Getting platforms!" << endl;
		return FAILURE;
	}

	/*For clarity, choose the first available platform. */
	if(numPlatforms > 0)
	{
		cl_platform_id* platforms = (cl_platform_id* )malloc(numPlatforms* sizeof(cl_platform_id));
		status = clGetPlatformIDs(numPlatforms, platforms, NULL);
		platform = platforms[0];
		free(platforms);
	}

	/*Step 2:Query the platform and choose the first GPU device if has one.Otherwise use the CPU as device.*/
	cl_uint				numDevices = 0;
	cl_device_id        *devices;
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);	
	if (numDevices == 0)	//no GPU available.
	{
		cout << "No GPU device available." << endl;
		cout << "Choose CPU as default device." << endl;
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);	
		devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
	}
	else
	{
		devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
	}
	

	/*Step 3: Create context.*/
	cl_context context = clCreateContext(NULL,1, devices,NULL,NULL,NULL);
	
	/*Step 4: Creating command queue associate with the context.*/
	cl_command_queue commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);

	/*Step 5: Create program object */
	const char *filename = "HelloWorld_Kernel.cl";
	string sourceStr;
	status = convertToString(filename, sourceStr);
	const char *source = sourceStr.c_str();
	size_t sourceSize[] = {strlen(source)};
	cl_program program = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);
	
	/*Step 6: Build program. */
	status=clBuildProgram(program, 1,devices,NULL,NULL,NULL);

	/*Step 7: Initial input,output for the host and create memory objects for the kernel*/
	const char* input = "GdkknVnqkc";
	size_t strlength = strlen(input);
	cout << "input string:" << endl;
	cout << input << endl;
	char *output = (char*) malloc(strlength + 1);
	double inputData[InputN*datanum];		// input layer
	double outputData[OutN*datanum];         // output layer
	double w_result[HN*InputN];
	double v_result[HN*OutN];
	double w[InputN][HN];		// weights from input layer to hidden layer
	double v[HN][OutN];			// weights from hidden layer to output layer

	double errors[1];
	
	// Generate data samples
	
	//Apertura de archivo
	//srand(time(NULL));

#pragma warning(disable:4996)
	FILE *fp;
	if ((fp = fopen("test2.txt", "r")) == NULL) {
		printf("Cannot open file.\n");
		exit(1);
	}

	int i = 0,j;
	int entradas[9][208], salidas[208];
	while (fscanf(fp, "%d %d %d %d %d %d %d %d %d %d", &entradas[0][i], &entradas[1][i], &entradas[2][i], &entradas[3][i], &entradas[4][i], &entradas[5][i], &entradas[6][i], &entradas[7][i], &entradas[8][i], &salidas[i]) != EOF) {
		//printf("%d %d %d %d %d %d %d %d %d %d", entradas[0][i], entradas[1][i], entradas[2][i], entradas[3][i], entradas[4][i], entradas[5][i], entradas[6][i], entradas[7][i], entradas[8][i], salidas[i]);
		//printf("\t%i\n", i);
		i++;
	}

	for (int i = 0; i < datanum; i++) {
		for (int j = 0; j < InputN; j++) {
			inputData[j+i*InputN] = (double)entradas[j][i];
			//printf("%f\n", inputData[j + i*j]);
		}
	}
		
	for (int i = 0; i < OutN*datanum; i++)
		outputData[i] = (double)salidas[i]/9;

	// Initializition
	for (i = 0; i<InputN*HN; i++) {
			w_result[i] = ((double)rand() / 32767.0) * 2 - 1;
	}
	for (i = 0; i<HN*OutN; i++) {
			v_result[i]= ((double)rand() / 32767.0) * 2 - 1;
	}
	for (int i = 0; i<40; i++) {
		for (int j = 0; j<9; j++) {
			//printf("got it %i,%i w:%2.1f\n", i, j, w_result[i * 9 + j]);
			w[j][i] = w_result[i * 9 + j];
		}
		//printf("got it %i v:%2.1f\n", i, v_result[i]);
		v[i][0] = v_result[i];
	}
	cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, (strlength + 1) * sizeof(char),(void *) input, NULL);
	cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY , (strlength + 1) * sizeof(char), NULL, NULL);
	cl_mem input2Buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, InputN * datanum * sizeof(double), (void *)inputData, NULL);//Entrada
	cl_mem input3Buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, OutN * datanum * sizeof(double), (void *)outputData, NULL);//Salida
	cl_mem input4Buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, HN * InputN  * sizeof(double), (void *)w_result, NULL);//Salida
	cl_mem input5Buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, HN * OutN  * sizeof(double), (void *)v_result, NULL);//Salida
	cl_mem output2Buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,  HN * InputN * sizeof(double), NULL, NULL);//Pesos Resultantes w
	cl_mem output3Buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, HN * OutN * sizeof(double), NULL, NULL);//Pesos Resultantes v
	cl_mem output4Buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 1*sizeof(double), NULL, NULL);//Error resultante

	/*Step 8: Create kernel object */
	cl_kernel kernel = clCreateKernel(program,"helloworld", NULL);

	/*Step 9: Sets Kernel arguments.*/
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&inputBuffer);
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&outputBuffer);
	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&input2Buffer);
	status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&input3Buffer);
	status = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&input4Buffer);
	status = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&input5Buffer);
	status = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&output2Buffer);
	status = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&output3Buffer);
	status = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *)&output4Buffer);

	starttime = omp_get_wtime();
	/*Step 10: Running the kernel.*/
	size_t global_work_size[1] = {strlength};
	status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);

	/*Step 11: Read the cout put back to host memory.*/
	status = clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0, strlength * sizeof(char), output, 0, NULL, NULL);
	status = clEnqueueReadBuffer(commandQueue, output2Buffer, CL_TRUE, 0, HN * InputN* sizeof(double), w_result, 0, NULL, NULL);
	status = clEnqueueReadBuffer(commandQueue, output3Buffer, CL_TRUE, 0, HN * OutN * sizeof(double), v_result, 0, NULL, NULL);
	status = clEnqueueReadBuffer(commandQueue, output4Buffer, CL_TRUE, 0, 1*sizeof(double), errors, 0, NULL, NULL);
	stoptime = omp_get_wtime();
	output[strlength] = '\0';	//Add the terminal character to the end of output.
	cout << "\noutput string:" << endl;
	cout << output << endl;

	for (int i = 0; i<40; i++) {
		for (int j = 0; j<9; j++) {
			//printf("got it %i,%i w:%2.1f\n", i, j, w_result[i * 9 + j]);
			w[j][i] = w_result[i * 9 + j];
		}
		//printf("got it %i v:%2.1f\n", i, v_result[i]);
		v[i][0] = v_result[i];
	}
	//printf("Error conseguido: %f \n", errors[0]);

	
	

	/*Step 12: Clean the resources.*/
	status = clReleaseKernel(kernel);				//Release kernel.
	status = clReleaseProgram(program);				//Release the program object.
	status = clReleaseMemObject(inputBuffer);		//Release mem object.
	status = clReleaseMemObject(outputBuffer);
	status = clReleaseCommandQueue(commandQueue);	//Release  Command queue.
	status = clReleaseContext(context);				//Release context.

	if (output != NULL)
	{
		free(output);
		output = NULL;
	}

	if (devices != NULL)
	{
		free(devices);
		devices = NULL;
	}

	
	

	double sigmoid(double);
	double x_out[InputN];		// input layer
	double hn_out[HN];			// hidden layer
	double y_out[OutN];         // output layer
	
	
	double max, min;
	double sumtemp;
	double errtemp;

	
	printf("\nError final: %f\n", errors[0]);
	etime = stoptime - starttime;
	wtick = omp_get_wtick();// presicion del contador
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

	printf("Ingresa 9 valores: ");
	scanf("%d %d %d %d %d %d %d %d %d", &entradas[0][0], &entradas[1][0], &entradas[2][0], &entradas[3][0], &entradas[4][0], &entradas[5][0], &entradas[6][0], &entradas[7][0], &entradas[8][0]);
	printf("Ingresaste: %d %d %d %d %d %d %d %d %d\n", entradas[0][0], entradas[1][0], entradas[2][0], entradas[3][0], entradas[4][0], entradas[5][0], entradas[6][0], entradas[7][0], entradas[8][0]);

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

	return SUCCESS;
}
// sigmoid serves as avtivation function
double sigmoid(double x) {
	return(1.0 / (1.0 + exp(-x)));
}