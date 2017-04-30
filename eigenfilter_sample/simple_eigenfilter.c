 #include <stdio.h>
#include <stdlib.h>
#include "./rng/rng.c"


// This code is a horribly unstylish.
// Use it ONLY as a straw man illustration of how to make GHA work at all, not
// as an example of how to do it well.


// Remember, in GHA, a layer of neurons is trained with a modified Hebbian learning algorithm.  The number
// of neurons in that single layer of neurons represents the number of principle components (dimensions of 
// transformed data points) that you wish to keep in the new representation.  The WEIGHTS of each of those 
// neurons ARE (after training) the Eigenvectors of the covariance matrix of the training data.  The weight 
// vector of neuron 1 (or zero if you're using a start at zero counting system) is the first principle 
// component.  The weights of the next neuron are collectively the Eigenvector of the second... and so on.
// In this paradigm, we train the network, but don't really use if for anything other than stripping off 
// the weight vectors and then using them as a coordinate transform.  It's a decentralized, low resource
// way of computing a PCA transform without having to explicitly compute covariance matrices and take 
// eigenvectors of same.  Awesome that....



#define X_n 2   // This is the dimensionality of the input pattern. 
                // It is also the number of inputs into each neuron.

#define Y_n 2   // This is the dimensionality of the output pattern.  
                // It is also the number of neurons in the output layer


int main()

{ double x[X_n];        // An array to hold the input patterns we're training on at any given time
  double y[Y_n];        // An array to hold the neuron outputs at any given time
  double w[Y_n][X_n];   // The weight matrix stored in the textbook's style.  W[j][i] is the weight FROM 
                        // input element i TO neuron j.  In this format, the Eigenvectors will appear in the
                        // FIRST dimension of the two dimensional array.  (All the i's of a setting of a j are
                        // the Eigenvector associated with neuron j.  One may order the stack of eigenvectors
                        // to form the PCA transform matrix in a straigtforward way.
                        
                        
  double dw[Y_n][X_n];  // This is just a scratch space matrix to hold delta weights computed by the modified
                        // Hebian rule.  No reason to break it out this way except to keep the syntax a little 
                        // more simple down the road at the cost of code efficiency.  Remember, I didn't say 
                        // this is an _awesome_ implementaiton
                        
                        
  double foo;           // An accumulator to hold the summation of 'k' neurons in the modified Hebbian rule
                        // see the textbook.  That weird little summation over k inside equation 8.80?  That's
                        // what we're stuffing in variable foo.  This is also done to keep the syntax a little more
                        // simple later.
                        
                        
  double learning_rate = 0.0001; // The learning rate.  I did this as a variable in case anyone wants to play with 
                                // variable learning rates as generally anticipated by the proofs underpinning
                                // this whole mess.  Alternatively, set it small and don't worry about it.  We'll
                                // talk about the effect of learning rate later in this code or in programming notes 
                                // that I send with it.  Not sure yet ;)

  int i,j,k;  // Generic counting variable we'll use when looping over input elements (i), neurons (j), and that
              // correction summation inside of equation 8.80 (k)
              
              
  int presentation;  // A counter to keep track of presentations of inputs to the network
  
  int sample;        // I'm doing random selection of patterns from the database.  This will be a holder for the
                     // index of the randomly selected pattern.
                     
  // The following are two sets of training data.  Both are IDENTICAL data in slightly different forms.  Consider the first to be "raw data"
  // that represents something someone collected.  The second is that raw data with element wise averages subtracted off (zero centered).  
  // Classic PCA requires zero centered data to function.  GHA does not, though it may be a good idea to do it anyway.  Both of these 
  // data sets are taken from one of the PCA tutorials I sent you (principle_components.pdf) and have relinked in my last message to you.
  // It stands to reason that GHA SHOULD get the same results as given in that tutorial.  We'll test this code by looking for that.  You can 
  // test your code this way too ;)
  
  // Oh... only uncomment one of these.....
  
  
//  Uncomment next two lines for raw data                                    
//  double TRAINING_DATA[10][2] = { {2.5, 2.4}, {0.5, 0.7}, {2.2, 2.9}, {1.9, 2.2}, {3.1, 3.0},
//                                  {2.3, 2.7}, {2.0, 1.6}, {1.0, 1.1}, {1.5, 1.6}, {1.1, 0.9}};

//  Uncomment next two lines for centered data
  double TRAINING_DATA[10][2] = { {0.69, 0.49}, {-1.31, -1.21}, {0.39, 0.99}, {0.09, 0.29}, {1.29, 1.09},
                                  {0.49, 0.79}, { 0.19, -0.31}, {-0.81, -0.81}, {-0.31, -0.31}, {-0.71, -1.01}};


  RNG *rng_one;  // Random number generator struct.  See rnd.c if you really need to know more (thank you eric for your fixes to rng.c, 
                 // I didn't put them in here yet, but I will at some point).

  
// The following code initializes the weight matrix.  One of the conditions of convervence of GHA is that one starts with 
// small random weights.  Investigate for yourself what happens if this condition is violated....  really... it's GOOD
// for you.


  rng_one = rng_create();  // Make a random number generator... see rnd.c if you really care.
  
  // Set those random weights, this should be fairly obvious
  for (j=0; j<Y_n; j++)
      for (i=0; i<X_n; i++)
         w[j][i] = rng_uniform(rng_one, -0.01, 0.01);



// Now, generally one would terminate training when one sees weights not changing very much anymore.  I'm being lazy and just
// hard coding a set number of presentations.  If you wanted to do something more sophisticated, you might monitor the delta weights
// and when the largest of them is under a certain threshold, call it quits.

for (presentation = 0; presentation < 10000000; presentation++)
{
  // Randomly select something from the training set and copy it into the input units.
  // The hard coded copying is hackalicious and wrong.  Really, I can't write a looping structure?
  
  sample = (int)trunc(rng_uniform(rng_one, 0.0, 9.999999));
  x[0] = TRAINING_DATA[sample][0];
  x[1] = TRAINING_DATA[sample][1];
  

  // This code computes the outputs of all the neurons in the system and places those numbers into
  // the y[] vector.  Remember, we really don't ultimately use these outputs for anything other than
  // fodder for the Hebbian learning.  We actually read our answers off the weight vectors....
  
  for (j=0; j<Y_n; j++)
      { y[j] = 0.0;
        for (i=0; i<X_n; i++)
           {y[j] += x[i] * w[j][i];
           }
      }

  // Here we compute the delta weights according to equation 8.8.  Now... HERE there is a little bit 
  // of sticky business.  This is basically a direct implementation of 8.8.  Note however that the count on
  // k is to j+1.  Think about this and figure out why I did it.  I imagine that at least some of the people
  // who had non-converging solutions counted only to J.  Think about the "differences" between 
  // summation notation and programatic looping constructs....

  for (j=0; j<Y_n; j++)
      { 
        for (i=0; i<X_n; i++)
            { foo = 0.0;
              for (k=0; k<(j+1); k++)
                  foo += w[k][i]*y[k];
              dw[j][i] = learning_rate * y[j]*(x[i] - foo);
            }
       }

   // Apply delta weights... just add the delta weights onto the weight matrix
   for (j=0; j<Y_n; j++)
       for (i=0; i<X_n; i++)
           w[j][i] += dw[j][i];

}

   // Print out eigenvectors in COLUMNS.  The raw weight matrix puts them in rows, I'm putting
   // them in columns to match the conventions of the tutorial document.
   
     for (i=0; i<X_n; i++)
       { for (j=0; j<Y_n; j++)
             printf("%lf  ", w[j][i]);
         printf("\n");
       }



}


       
