#ifndef CPULAYERS_HPP
#define CPULAYERS_HPP
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <random>

using namespace std;


template<typename T>
struct CPUNoneAct
{
  inline static void  compute( T* __restrict__  inp, T* __restrict__ out)
  {
      *out = *inp ;
  }
};

template<typename T>
struct CPURelu
{
  inline static void compute( T* __restrict__  inp, T* __restrict__ out)
  {
      *out = *inp > 0.0 ? *inp:0.0;
  }
};

template<typename T>
struct CPUSelu
{
  inline static void compute( T* __restrict__  inp, T* __restrict__ out)
  {
      const T alpha = 1.67326324;
      const T scale = 1.05070098;
      *out = *inp > 0.0 ? scale * (*inp) : scale * alpha * ( exp(*inp ) - 1) ;
  }
};

template<typename T, typename TAct>
struct CPUDenseLayer
{
  T* __restrict__  W;
  T* __restrict__  b;
  int inputDim;
  int outputDim;

  void allocate()
  {
      this->W = new T[inputDim*outputDim];
      this->b = new T[outputDim];
  }

  template<typename TRng>
  void init( TRng* e2)
  {
      int wsize = inputDim*outputDim ;
      int bsize = outputDim;
      //Glorot_uniform initialization for W and 0 for b
      double limit = sqrt(6.0 / (inputDim + outputDim));
      std::uniform_real_distribution<> dist(-limit, limit);

      for( int i = 0 ; i <wsize ; i++)
          W[i] = dist(*e2) ;

      for( int i = 0 ; i < bsize ; i++)
          b[i] = 0.0;

  }

  void setZero()
  {
    int wsize = inputDim*outputDim ;
    int bsize = outputDim;
    for( int i = 0 ; i <wsize ; i++)
        W[i] = 0.0 ;
    for( int i = 0 ; i < bsize ; i++)
        b[i] = 0.0;
  }

  void increment( T scale, CPUDenseLayer<T,TAct>* params )
  {
    int wsize = inputDim*outputDim ;
    int bsize = outputDim;
    for( int i = 0 ; i <wsize ; i++)
        W[i] += params->W[i] * scale ;
    for( int i = 0 ; i < bsize ; i++)
        b[i] += params->b[i] * scale ;
  }



};

//Extracted from CPUDenseLayer class to make sure there is __restrict__ on what would be this
template<typename T, typename TAct>
void computeDense( const CPUDenseLayer<T,TAct>& __restrict__ layer, T*__restrict__  p, T*__restrict__  out, T* __restrict__  outnoact )
{
  const int n = layer.outputDim;
  const int m = layer.inputDim;

  const double *__restrict__ W =  layer.W;
  const double *__restrict__ b =  layer.b;

  for( int i = 0 ; i < n ; i++)
  {
    T temp = 0.0;
    for( int j = 0 ; j < m ; j++)
    {
       temp +=  W[i*m+j]*p[j];
    }
    temp += b[i];
    outnoact[i] = temp;
  }
  for( int i = 0 ; i < n ; i++)
  {
      TAct::compute(&outnoact[i], &out[i]);
  }
}


template<typename T>
T sparseCategoricalCrossEnropy( T* logit, int n,  int label )
{
  //exp(logits) / sum(exp(logits), axis=-1)
  T sumexp = 0.0;
  for( int i = 0 ; i < n ; i++)
  {
    sumexp += exp(logit[i]);
  }
  //return -log( exp( logit[label]) / sumexp );
  return - ( logit[label] - log( sumexp ) );
}

template <typename T>
int argmax( T* v, int n)
{
  T m = v[0];
  int am = 0;
  for( int i = 1 ; i < n ; i++)
  {
      if( v[i] > m )
      {
        am = i;
        m = v[i];
      }
  }
  return am;
}



#endif
