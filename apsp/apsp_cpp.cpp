/**
 * Benchmarking program written to compare the performance of 
 * different All-Pairs-Shortest-Paths implementations:
 * 	1) Standard Iterative Algorithm
 *	2) In-Place Recursive Implementation
 *	 (Multithreaded version using Cilk++)

 *	Possible additions:
 *	1) Sparse Matrix Based Recursive Implementation
 *	2) Sparse Matrix Based Iterative Implementation
 *  	3) Blocked Iterative Implementation
 *
 * 09/23/2008 - Aydin Buluc - aydin@cs.ucsb.edu
 * 11/04/2008 - John Hart   - jhart@cilk.com
 *  This file is built with Cilk++ g++ to produce the serial version

**/

#include <cilk.h>
//#include "cilk_util.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <limits>
#include <stdio.h>
#include <stdlib.h>

// Threshold to fall back to iterative i-j-k multiplication
#define THRESHOLD	8

// Serialization
#define cilk_spawn 
#define cilk_sync
#define cilk_for for	

using namespace std;

void fw(float**, int,int,int);
void fw_alt(float**, int,int,int);
void sub_mm_add(float**,float**,float**,int,int,int,int,int,int,int,int,int);
void sub_mm_alt(float**,float**, float**, int, int, int, int,int, int, int, int);


int ** pred;	// global predecessor matrix

template <typename T>
struct inf_plus{
  T operator()(const T& a, const T& b) const {
    T inf = std::numeric_limits<T>::max();
    if (a == inf || b == inf){
      return inf;
    }
    return a + b;
  }
};


void printMat(float ** matrix, int width, int height, ofstream & output)
{
	float float_inf = std::numeric_limits<float>::max();
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			if( matrix[i][j] != float_inf)
			{
				output << i <<"\t" << j << "\t" << matrix[i][j] << endl;
			}
		}
	}
}

void printDiff(float ** iterative, float ** recursive, int width, int height)
{
  int error_count=0;
  for (int i=0; i<height; i++) 
  {
    for (int j=0; j<width; j++) 
	{
      if ( abs(iterative[i][j] - recursive[i][j]) > 0.001) 
	  {
			error_count++;
			printf("iterative[%d][%d] = %f\n", i,j, iterative[i][j]);
			printf("recursive[%d][%d] = %f\n", i,j, recursive[i][j]);
      }
    }
  }
  printf("\nTotal Errors = %d\n", error_count);
}

void LoadMatrix(ifstream & input, float ** distMatrix, int size)
{
	int read = 0;
	int v1, v2;
	float value;

	if (input.is_open())
	{
		while (! input.eof() && read < size)
		{
			input >> v1 >> v2 >> value;
			v1--;
			v2--;

			distMatrix[v1][v2] = fabs(value);	// no negative edges for now
			read++;
		}
	}
}

void computeGoldI(float ** C,  int hA)
{
    inf_plus<float> combine;

    for ( int k = 0; k < hA; ++k)
		cilk_for ( int i = 0; i < hA; ++i)
			for ( int j = 0; j < hA; ++j)
				C[i][j] = min (C[i][j], combine(C[i][k],C[k][j]));
}

void computeGoldR(float ** C, int hA)
{
    fw(C,0,0,hA);
}


void fw(float ** A, int sX, int sY, int h)
{
    if(h > 64)
    {
        int size  = h/2;	// new size

	//a = a*
	fw(A, sX, sY, size);
        
	// All the calls to sub_mm have di=dj=dk=size as the last three parameters
	// as the calls made from fw are always for square matrix multiplication
 
	//b = ab
	cilk_spawn sub_mm_add(A,A,A,sX,sY+size,sX,sY,sX,sY+size,size,size,size);
        
	//c = ca
	sub_mm_add(A,A,A,sX+size,sY,sX+size,sY,sX,sY,size,size,size);
	
	cilk_sync;
	
	// d = d + cb;
	sub_mm_add(A,A,A,sX+size,sY+size,sX+size,sY,sX,sY+size,size,size,size);
       
	//d=d*
        fw(A,sX+size,sY+size, size);
        
        //b = bd
        cilk_spawn sub_mm_add(A,A,A,sX,sY+size,sX,sY+size,sX+size,sY+size,size,size,size);
       
	//c = dc
        sub_mm_add(A,A,A,sX+size,sY,sX+size,sY+size,sX+size,sY,size,size,size);
       
        cilk_sync;
      
	// a = a + bc
        sub_mm_add(A,A,A,sX,sY,sX,sY+size,sX+size,sY,size,size,size); 

    }
	else	// h = 64 -> 64*64*4 = 16 KB where L1 data cache is 32 KB
	{
		inf_plus<float> combine;
		for ( int k = 0; k < h; ++k)
		{
			for ( int i = 0; i < h; ++i)
			{
				for ( int j = 0; j < h; ++j)
				{
					// A[i+sX][j+sY] = min (A[i+sX][j+sY], combine(A[i+sX][k+sY],A[k+sX][j+sY]));
					float nlen = combine(A[i+sX][k+sY],A[k+sX][j+sY]);
					if( nlen < A[i+sX][j+sY] )
					{
						A[i+sX][j+sY] = nlen;
						pred[i+sX][j+sY] = pred[k+sX][j+sY]; // sX == sY
					}
				}
			}
		}
	}
}

void fw_alt(float ** A, int sX, int sY, int h)
{
    if(h > 64)
    {
        int size  = h/2;	// new size

	//a = a*
	fw(A, sX, sY, size);

	//b = ab
	cilk_spawn sub_mm_alt(A,A,A,sX,sY+size,sX,sY,sX,sY+size,size, 0);
        
	//c = ca
	sub_mm_alt(A,A,A,sX+size,sY,sX+size,sY,sX,sY,size, 0);
	
	cilk_sync;
	
	// d = d + cb;
	sub_mm_alt(A,A,A,sX+size,sY+size,sX+size,sY,sX,sY+size,size, 0);
       
	//d=d*
        fw(A,sX+size,sY+size, size);
        
        //b = bd
        cilk_spawn sub_mm_alt(A,A,A,sX,sY+size,sX,sY+size,sX+size,sY+size,size, 0);
       
	//c = dc
        sub_mm_alt(A,A,A,sX+size,sY,sX+size,sY+size,sX+size,sY,size, 0);
       
        cilk_sync;
      
	// a = a + bc
        sub_mm_alt(A,A,A,sX,sY,sX,sY+size,sX+size,sY,size,  0); 

    }
	else	// h = 64 -> 64*64*4 = 16 KB where L1 data cache is 32 KB
	{
		inf_plus<float> combine;
		for ( int k = 0; k < h; ++k)
			for ( int i = 0; i < h; ++i)
				for ( int j = 0; j < h; ++j)
					A[i+sX][j+sY] = min (A[i+sX][j+sY], combine(A[i+sX][k+sY],A[k+sX][j+sY]));
	}
}
/**
  * Executes both C <- A * B and C <- C + A * B on the (min,+) semiring
  * Recursive with a branching factor of 8, works only on square matrices whose dimensions are powers of two
  * sXc: x-offset of matrix C
  * sYc: y-offset of matrix C
  * hA: dimensions of (all) matrices
  */
void sub_mm_alt(float ** C,float ** A, float ** B, int sXc, int sYc, int sXa, int sYa,int sXb, int sYb, int hA,  int depth)
{
	if(hA > THRESHOLD)
	{
		int h = hA/2;
	
		// First wave of updates 
		
		if(depth++ % 2 == 1)	// no sharing of A
		{

			cilk_spawn sub_mm_alt(C, A, B, sXc, sYc, sXa, sYa, sXb, sYb, h, depth); 		// Update C_11, access A_11 & B_11
			cilk_spawn sub_mm_alt(C, A, B, sXc, sYc+h, sXa, sYa+h, sXb+h, sYb+h, h,  depth); 	// Update C_12, access A_12 & B_22
			cilk_spawn sub_mm_alt(C, A, B, sXc+h, sYc, sXa+h, sYa, sXb, sYb, h, depth); 		// Update C_21, access A_21 & B_11
			sub_mm_alt(C, A, B, sXc+h, sYc+h, sXa+h, sYa+h, sXb+h, sYb+h, h, depth); 		// Update C_22, access A_22 & B_22

			cilk_sync;

			// Second wave of updates
			cilk_spawn sub_mm_alt(C, A, B, sXc, sYc, sXa, sYa+h, sXb+h, sYb, h,  depth); 		// Update C_11, access A_12 & B_21
			cilk_spawn sub_mm_alt(C, A, B, sXc, sYc+h, sXa, sYa, sXb, sYb+h, h, depth); 		// Update C_12,	access A_11 & B_12
			cilk_spawn sub_mm_alt(C, A, B, sXc+h, sYc, sXa+h, sYa+h, sXb+h, sYb, h, depth);  	// Update C_21, access A_22 & B_21
			sub_mm_alt(C, A, B, sXc+h, sYc+h, sXa+h, sYa, sXb, sYb+h, h,  depth); 			// Update C_22, access A_21 & B_12

			cilk_sync;
		}
		else	// no sharing of B
		{
			cilk_spawn sub_mm_alt(C, A, B, sXc, sYc, sXa, sYa, sXb, sYb, h,  depth); 		// Update C_11, access A_11 & B_11
			cilk_spawn sub_mm_alt(C, A, B, sXc, sYc+h, sXa, sYa, sXb, sYb+h, h,  depth); 		// Update C_12,	access A_11 & B_12
			cilk_spawn sub_mm_alt(C, A, B, sXc+h, sYc, sXa+h, sYa+h, sXb+h, sYb, h,  depth);  	// Update C_21, access A_22 & B_21
			sub_mm_alt(C, A, B, sXc+h, sYc+h, sXa+h, sYa+h, sXb+h, sYb+h, h,  depth); 		// Update C_22, access A_22 & B_22

			cilk_sync;

			// Second wave of updates
			cilk_spawn sub_mm_alt(C, A, B, sXc, sYc, sXa, sYa+h, sXb+h, sYb, h,  depth); 		// Update C_11, access A_12 & B_21
			cilk_spawn sub_mm_alt(C, A, B, sXc, sYc+h, sXa, sYa+h, sXb+h, sYb+h, h,  depth); 	// Update C_12, access A_12 & B_22
			cilk_spawn sub_mm_alt(C, A, B, sXc+h, sYc, sXa+h, sYa, sXb, sYb, h,  depth); 		// Update C_21, access A_21 & B_11
			sub_mm_alt(C, A, B, sXc+h, sYc+h, sXa+h, sYa, sXb, sYb+h, h,  depth);	 		// Update C_22, access A_21 & B_12

			cilk_sync;
		}
		
	}
	else
	{
		// The problem is now small enough that we can just do things serially (in an iterative fashion).
		inf_plus<float> combine;
		for (int i = 0; i < hA; ++i) 
		{
  			for (int j = 0; j < hA; ++j) 
			{
				float minij = C[sXc+i][sYc+j];
   				for (int k = 0; k < hA; ++k)
				{
    					minij = min (minij, combine(A[sXa+i][sYa+k],B[sXb+k][sYb+j]));
     				} 

				C[sXc+i][sYc+j] = minij;
			}
		}
	}
}


/**
  * Executes C <- C + A * B on the (min,+) semiring
  * sXc: x-offset of matrix C
  * sYc: y-offset of matrix C
  * di: length of the i dimension of GEMM operation
  * ABAB: This is a very general GEMM, for the APSP operations note that
  * 	1- There is only one matrix, A=B=C
  * 	2- sYa = sXb so that A[][sYa+k] B[sXb+k][] is a path 
  */
void sub_mm_add(float ** C,float ** A, float ** B, int sXc, int sYc, int sXa, int sYa,int sXb, int sYb, int di, int dj, int dk)
{
    if (di >= dj && di >= dk && di > THRESHOLD) 
    	{
		cilk_spawn sub_mm_add(C, A, B, sXc, sYc, sXa, sYa, sXb, sYb, di/2, dj, dk);
		sub_mm_add(C, A, B, sXc+di/2, sYc, sXa+di/2, sYa, sXb, sYb, di-(di/2), dj, dk);
		cilk_sync;
	}	
	else if (dj >= dk && dj > THRESHOLD) 
	{
		cilk_spawn sub_mm_add(C, A, B, sXc, sYc, sXa, sYa, sXb, sYb, di, dj/2, dk); 
		sub_mm_add(C, A, B, sXc, sYc+dj/2, sXa, sYa, sXb, sYb+dj/2, di, dj-(dj/2), dk); 
		cilk_sync;
	} 
	else if (dk > THRESHOLD) 
	{
		// It's not safe to use a spawn here because both of the recursive calls are 
		// updating the same matrix block -> race condition 
		sub_mm_add(C, A, B, sXc, sYc, sXa, sYa, sXb, sYb, di, dj, dk/2);
		sub_mm_add(C, A, B, sXc, sYc, sXa, sYa+dk/2, sXb+dk/2, sYb, di, dj, dk-dk/2);
	}
	else 
	{
		// The problem is now small enough that we can just do things serially (in an iterative fashion).
		inf_plus<float> combine;
    
		for (int i = 0; i < di; ++i) 
		{
  			for (int j = 0; j < dj; ++j) 
			{
				float minij = C[sXc+i][sYc+j]; 
				int vij = pred[sXc+i][sYc+j];
   				for (int k = 0; k < dk; ++k)
				{
					// minij = min (minij, combine(A[sXa+i][sYa+k],B[sXb+k][sYb+j]));
					float nlen = combine(A[sXa+i][sYa+k],B[sXb+k][sYb+j]);
					if( nlen < minij )
					{
						minij = nlen;
						vij = pred[sXb+k][sYb+j];	// sYa == sXb 
					}
				}
				C[sXc+i][sYc+j] = minij;
				pred[sXc+i][sYc+j] = vij;
			}
		}
	}	
}


// prints back wards
void print_path (float ** distmatrix, int i, int j) 
{
  if (i!=j && pred[i][j] !=j) {
        cout << "From " << pred[i][j] << " to " << j << " with distance " << distmatrix[pred[i][j]][j] << endl;
        print_path(distmatrix, i,pred[i][j]);
  }
}

#ifndef WIN32
#define cilk_main cilk_main_proxy
#endif

int cilk_main(int argc, char *argv[])
{
	ifstream input("matrix.txt");

	int m,n, nnz;
	input >> m >> n >> nnz;

	std::less<float> compare;
    	inf_plus<float> combine;
	float float_inf = std::numeric_limits<float>::max();

	float ** distMatrix;
	float ** goldMatrix;

	distMatrix = new float*[m];
	goldMatrix = new float*[m];
	pred = new int*[m];
	for(int i=0; i< m; i++)
	{
		distMatrix[i] = new float[n];
		goldMatrix[i] = new float[n];
		pred[i] = new int[n];
		for(int j=0; j< n; j++)
		{
			distMatrix[i][j] = float_inf;
			pred[i][j] = i;
		}
	}

	LoadMatrix(input, distMatrix, nnz);

	for(int i=0; i< m; i++)
	{
		distMatrix[i][i] = 0;	// diagonals should be zero
	}

	// Copy distMatrix to goldMatrix as input
    	for ( int i = 0; i < m; ++i)
		for ( int j = 0; j < m; ++j)
			goldMatrix[i][j] = distMatrix[i][j];

	long t1 = 0.0;//cilk_get_time();
	computeGoldI(goldMatrix, m);
	long t2 = 0.1;//cilk_get_time();
	cout<<"Iterative: "<< (t2-t1)/1000 <<"."<<(t2-t1)%1000 <<" seconds" <<endl;

	float ** recuMatrix = new float*[m];
	for(int i=0; i< m; i++)
	{
		recuMatrix[i] = new float[n];
	}
	// Copy distMatrix to recuMatrix as input
    	for ( int i = 0; i < m; ++i)
		for ( int j = 0; j < m; ++j)
			recuMatrix[i][j] = distMatrix[i][j];

	t1 = 0.0;//cilk_get_time();
	computeGoldR(recuMatrix, m);
	t2 = 0.1;//cilk_get_time();

	cout<<"Recursive: "<< (t2-t1)/1000 <<"."<<(t2-t1)%1000 <<" seconds" <<endl;

//	printDiff(goldMatrix,recuMatrix, m, n);

	ofstream outputr("recursive.txt"); 
	printMat(recuMatrix, m, n, outputr);
	outputr.close();

//	ofstream outputi("iterative.txt"); 
//	printMat(goldMatrix, m, n, outputi);
//	outputi.close();
	
	while(true)
	{
		string strfrom, strto;
		cout << "Enter from vertex (-1 to finish)...";
		cin >> strfrom;
		if(strfrom == "-1")
			break;
		cout << "Enter to vertex...";
		cin >> strto;

		int from = atoi(strfrom.c_str());
		int to = atoi(strto.c_str());
		print_path (distMatrix, from, to);
	}

	for(int i=0; i< m; i++)
	{
		delete [] distMatrix[i];
	}
	delete distMatrix;

	for(int i=0; i< m; i++)
	{
		delete [] goldMatrix[i];
	}
	delete goldMatrix;

	for(int i=0; i< m; i++)
	{
		delete [] recuMatrix[i];
	}
	return 0;
}

#ifndef WIN32
// Boiler plate code required for GCC
struct args_s
{
	int argc;
	char **argv;
};

int cilk_main0(void *args_)
{
	args_s *args = static_cast<args_s *>(args_);
	return cilk_main_proxy(args->argc, args->argv);
}

int main(int argc, char *argv[])
{
	int nWorkers = 0;
	args_s args;
	char* arg0 = argv[0];

	args.argc = argc;
	args.argv = argv;
//  This is the pure C++ version.
                    int retval = cilk_main_proxy (argc, argv);
//	cilk::context * ctx = cilk::create_context();
//	if (nWorkers)
//		ctx->set_worker_count(nWorkers);

//	int retval = ctx->run(cilk_main0, (void*) &args);
//	cilk::destroy_context(ctx);
	return retval;
}
#endif
