#include <vector>
#include <algorithm>
#include <iostream>
#include <unistd.h>
#include <random>

#include "mnist/mnist_reader.hpp"
#include "funzyme/cpulayers.hpp"

using namespace std;

extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_out;
extern int enzyme_const;

void __enzyme_autodiff(...);

class NetDefinition;

class NetMemory
{
public:
    NetMemory(  NetDefinition* nd ):nd(nd)
    {
    }
    void allocate();
    void setZero();

    NetDefinition* __restrict__ nd;
    double * __restrict__ f1;
    double * __restrict__ f2;
    double * __restrict__ f1NoAct;
    double * __restrict__ f2NoAct;
    double * __restrict__ outNoAct;
};

class NetDefinition
{
    public :
    NetDefinition(int inputDim, int featDim, int outputDim)
    {
        layer1.inputDim = inputDim;
        layer1.outputDim = featDim;
        layer2.inputDim = featDim;
        layer2.outputDim = featDim;
        outputlayer.inputDim = featDim;
        outputlayer.outputDim = outputDim;
    }

    void allocate()
    {
      layer1.allocate();
      layer2.allocate();
      outputlayer.allocate();
    }

    template <typename TRng>
    void init(TRng* rng)
    {
      layer1.init(rng);
      layer2.init(rng);
      outputlayer.init(rng);
    }

    void setZero()
    {
        layer1.setZero();
        layer2.setZero();
        outputlayer.setZero();
    }

    void increment( double scale, NetDefinition* params )
    {
        layer1.increment( scale, &params->layer1);
        layer2.increment( scale, &params->layer2);
        outputlayer.increment( scale, &params->outputlayer);
    }

    void compute( NetMemory*__restrict__  mem,  double* __restrict__  p, double* __restrict__  out )
    {
       computeDense<double,CPUSelu<double>>(layer1, p , mem->f1,mem->f1NoAct);
       computeDense<double,CPUSelu<double>>(layer2,mem->f1, mem->f2,mem->f2NoAct);
       computeDense<double,CPUNoneAct<double>>(outputlayer,mem->f2, out,mem->outNoAct);
    }

    CPUDenseLayer<double,CPUSelu<double>> layer1;
    CPUDenseLayer<double,CPUSelu<double>> layer2;
    CPUDenseLayer<double,CPUNoneAct<double>> outputlayer;
};

void NetMemory::allocate()
{
      f1 = new double[nd->layer1.outputDim ];
      f2 = new double[nd->layer2.outputDim ];
      f1NoAct = new double[nd->layer1.outputDim ];
      f2NoAct = new double[nd->layer2.outputDim ];
      outNoAct = new double[nd->outputlayer.outputDim ];
}

void NetMemory::setZero()
{
      for( int i = 0 ; i < nd->layer1.outputDim ; i++ )
      {
        f1[i] = 0.0;
        f1NoAct[i] = 0.0;
      }
      for( int i = 0 ; i < nd->layer2.outputDim ; i++ )
      {
        f2[i] = 0.0;
        f2NoAct[i] = 0.0;
      }
      for( int i = 0 ; i < nd->outputlayer.outputDim  ; i++ )
      {
          outNoAct[i] = 0.0;
      }
}

class NetWrapper
{
public:
  NetWrapper(int inputDim, int featDim, int outputDim )
  {
    nd = new NetDefinition( inputDim, featDim, outputDim);
    nm = new NetMemory(nd);
    nd->allocate();
    nm->allocate();
    this->inputDim = inputDim;
    this->outputDim = outputDim;
    output = new double[outputDim];
    input = new double[inputDim];
  }

  void setZero()
  {
    for( int i = 0 ; i < this->outputDim ; i++)
      output[i] = 0.0;
    for( int i = 0 ; i < this->inputDim ; i++)
      input[i] = 0.0;
    nm->setZero();
    nd->setZero();
  }

  NetDefinition* __restrict__ nd;
  NetMemory* __restrict__ nm;
  double* __restrict__ input;
  double* __restrict__ output;
  int inputDim;
  int outputDim;
};

void netLoss( NetWrapper* __restrict__  nw, int label,  double* __restrict__  out )
{
  nw->nd->compute(nw->nm, nw->input, nw->output );
  *out = sparseCategoricalCrossEnropy<double>(nw->output, nw->nd->outputlayer.outputDim, label);
}

void train( const mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t>& dataset )
{
      NetWrapper nw( 784,100,10);
      NetWrapper dnw( 784,100,10);

      std::mt19937 rng(42);
      nw.nd->init(&rng);

      cout << "net initialized " << endl;
      double lr= 1e-4;
      int nbepoch = 3;

      std::vector<int> perm(dataset.training_images.size());
      std::iota(perm.begin(), perm.end(), 0);

      for( int k = 0; k < nbepoch ; k++)
      {
        cout << "Starting epoch : " << k << endl;
        std::shuffle(perm.begin(), perm.end(), rng);
        double sumloss = 0.0;
      for( int i = 0; i < dataset.training_images.size() ; i++)
      {

          const vector<uint8_t>& img = dataset.training_images[perm[i]];
          for( int j = 0 ; j < img.size() ; j++)
          {
            nw.input[j] = ((double) img[j] - 128 ) / 128.0;
          }
          double loss = 0.0;
          double dloss = 1.0;
          //netLoss(&nw, dataset.training_labels[i],  &loss  );
          dnw.setZero();
          int label = dataset.training_labels[perm[i]];
          __enzyme_autodiff( netLoss, enzyme_dup, &nw, &dnw,
                                                          enzyme_const, label,
                                                          enzyme_dup, &loss, &dloss);
          nw.nd->increment(-lr,dnw.nd);
          //cout << "loss : " << loss << endl;
          sumloss += loss;
      }
      cout << "average epoch loss " << sumloss / dataset.training_images.size() << endl;
    }

    cout << "training done " << endl;
    cout << "testing " << endl;
    //Do the prediction on the test set
    int score = 0;
    for( int i = 0; i < dataset.test_images.size() ; i++)
    {
        const vector<uint8_t>& img = dataset.test_images[i];
        for( int j = 0 ; j < img.size() ; j++)
        {
          nw.input[j] = ((double) img[j] - 128 ) / 128.0;
        }
        nw.nd->compute(nw.nm, nw.input, nw.output);
        int pred = argmax( nw.output, 10 );
        int label = dataset.test_labels[i];
        //cout << "pred " << pred << " label " << label << endl;
        if( pred == label )
        {
          score++;
        }
    }

    double accuracy = (double)score / dataset.test_images.size();
    cout << "Accuracy : " << accuracy << endl;

}

int main(int argc, char** argv )
{
cout << "testMnist " << endl;

mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
       mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("./data");

   std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
   std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
   std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
   std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

   cout << "Before training" << endl;
   train( dataset);
   cout << "After training" << endl;
}
