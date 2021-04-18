
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <algorithm>
#include <random>
#include <windows.h>

#define BLOCK_SIZE 32
#define MIN_VALUE -25
#define MAX_VALUE 25
#define MATRIX_SIZE 1000
#define DELTA 10e-10

using namespace std;


__global__ void matrixMult(double* __restrict__ A, double* __restrict__ B, double* __restrict__ C, size_t size)
{
	size_t i = blockDim.y * blockIdx.y + threadIdx.y;
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= size || j >= size)
		return;

	size_t ind = i * size + j;
	C[ind] = 0;
	for (size_t k = 0; k < size; k++) {
		C[ind] += A[i * size + k] * B[k * size + j];
	}

}

__global__ void matrixMultShared(double* __restrict__ A, double* __restrict__ B, double* __restrict__ C, size_t size)
{
	size_t tx = threadIdx.x;
	size_t ty = threadIdx.y;

	size_t i = blockDim.y * blockIdx.y + ty;
	size_t j = blockDim.x * blockIdx.x + tx;

	double cellValue = 0;

	__shared__ double a[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ double b[BLOCK_SIZE][BLOCK_SIZE];

	for (size_t k = 0; k * BLOCK_SIZE < size; k++) {

		if (i < size && k * BLOCK_SIZE + tx < size) {
			a[ty][tx] = A[i * size + k * BLOCK_SIZE + tx];
		}
		else {
			a[ty][tx] = 0;
		}

		if (j < size && k * BLOCK_SIZE + ty < size) {
			b[ty][tx] = B[(k * BLOCK_SIZE + ty) * size + j];
		}
		else {
			b[ty][tx] = 0;
		}

		__syncthreads();

		for (size_t l = 0; l < BLOCK_SIZE; l++) {
			cellValue += a[ty][l] * b[l][tx];
		}

		__syncthreads();
	}
	if (i < size && j < size) {
		C[i * size + j] = cellValue;
	}

}



float matrixMulOnGPU(double* A, double* B, double* C, size_t size, bool shared) {

	double* dA, * dB, * dC;

	size_t numBytes = size * size * sizeof(double);

	cudaEvent_t start, end;
	float time;

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks((size + threads.x - 1) / threads.x, (size + threads.y - 1) / threads.y);

	cudaMalloc((void**)(&dA), numBytes);
	cudaMalloc((void**)(&dB), numBytes);
	cudaMalloc((void**)(&dC), numBytes);

	cudaMemcpy(dA, A, numBytes, cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, numBytes, cudaMemcpyKind::cudaMemcpyHostToDevice);

	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start, 0);

	if (shared) {
		matrixMultShared << <blocks, threads >> > (dA, dB, dC, size);
	}
	else {
		matrixMult << <blocks, threads >> > (dA, dB, dC, size);
	}

	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time, start, end);

	cudaEventDestroy(start);
	cudaEventDestroy(end);

	cudaMemcpy(C, dC, numBytes, cudaMemcpyKind::cudaMemcpyDeviceToHost);

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
	return time / 1000.0f;
}

float matrixMulOnCPU(double* A, double* B, double* C, size_t size) {

	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency);

	LARGE_INTEGER start;
	QueryPerformanceCounter(&start);

	for (size_t i = 0; i < size; i++) {
		for (size_t j = 0; j < size; j++) {
			size_t ind = i * size + j;
			C[ind] = 0;
			for (size_t k = 0; k < size; k++) {
				C[ind] += A[i * size + k] * B[k * size + j];
			}
		}
	}

	LARGE_INTEGER end;
	QueryPerformanceCounter(&end);

	return (float)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
}



bool compareMatrices(double* A, double* B, size_t size) {
	size_t numCells = size * size;
	for (int i = 0; i < numCells; i++) {
		if (std::abs(A[i] - B[i]) > DELTA) {
			return false;
		}
	}
	return true;
}

double* generateRandomMatrix(size_t size)
{
	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<> distr(MIN_VALUE, MAX_VALUE);

	size_t numElements = size * size;
	double* matrix = new double[numElements];

	for (size_t i = 0; i < numElements; i++) {
		matrix[i] = distr(gen);
	}

	return matrix;
}

int main()
{
	setlocale(LC_ALL, "Russian");
	double* hA = generateRandomMatrix(MATRIX_SIZE);
	double* hB = generateRandomMatrix(MATRIX_SIZE);
	double* hCGPU = new double[MATRIX_SIZE * MATRIX_SIZE];
	double* hCGPU_shared = new double[MATRIX_SIZE * MATRIX_SIZE];
	double* hCCPU = new double[MATRIX_SIZE * MATRIX_SIZE];
	double time;

	cout << "Начато вычисление на GPU c Shared Memory..." << endl;
	time = matrixMulOnGPU(hA, hB, hCGPU_shared, MATRIX_SIZE, true);
	cout << "Вычисление на GPU завершено за " << time << "секунд" << endl;
	cout << "Начато вычисление на GPU без Shared Memory..." << endl;
	time = matrixMulOnGPU(hA, hB, hCGPU, MATRIX_SIZE, false);
	cout << "Вычисление на GPU завершено за " << time << "секунд" << endl;
	cout << "Начато вычисление на CPU..." << endl;
	time = matrixMulOnCPU(hA, hB, hCCPU, MATRIX_SIZE);
	cout << "Вычисление на CPU завершено за " << time << "секунд" << endl;
	cout << "Сравнение матриц:" << endl;
	if (compareMatrices(hCGPU, hCCPU, MATRIX_SIZE) && compareMatrices(hCGPU_shared, hCCPU, MATRIX_SIZE))
		cout << "Результаты вычислений совпадают!" << endl;
	else
		cout << "Результаты вычислений НЕ совпадают!" << endl;

	delete[] hA;
	delete[] hB;
	delete[] hCGPU;
	delete[] hCCPU;
	delete[] hCGPU_shared;
	return 0;
}
