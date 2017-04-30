The folder contains following subfolders:

1. eigenfilter_sample : this folder contains "simple_eigenfilter.c" source code which generates eigen values of given data. An example data is provided inside the source code implementation. The C file leverages "rng" library which is random number generator library.

I have already compiled the "C source code file"
execute it as follows:

On the terminal or Command Prompt
./a.out

2. Question1_SourceCode : this folder contains the implementation of Generalized Hebbian Learning Algorithm in python. The python source code file "hebbianlearning.py" is implementation of hebbian learning.
"C Source Code" files "epoch.c", "normalization.c" and "pattern.c" contains implementation of back-propagation and perceptron architecture. These are surplus code which I used for GHA. But later on implemented using Python.

3. Question2_SourceCode : this folder contains the implementation of Self Organization Map (mySOM.py), Application of SOM to "animals_from_jcg.dat". On the execution of SOM_self_computing4.py, you will get a 2-D lattice representation of data labels and a text file. The text file generated has the same format of ".cod" files. Hence can be used with som_pak package.

Pre-execution steps :


/********************* This snippet reads your .dat file ************************/
/*************** Our implementation is specific to the data file ***************/
Make your data similar to example "animal_from_jcg.dat" file.

This snippet is defined in the code. This snippet reads your file and separates the features and labels from the dataset.

with open('animals_from_jcg.dat') as input_file:
    for line in input_file:
        features.append(map(int,line.split('\t')[0:13]))
        labels.append((line.split('\t')[-1]).rstrip('\n'))


/************** This function initiates the self organizing map structure ****************/

som = MiniSom(10,10,13,sigma=1.0, learning_rate=0.02)

(10,10,13) is x dimension = 10, y dimension = 10 and the each weight vector is of dimension with size 13.
sigma=1.0 is the width the gaussian function taken as the neighborhood determination function
learning rate = 0.02 is the learning rate of the hebbian learning network.

execution details :

python SOM_self_computing4.py

4. Gaur_SoftComputing4_documentation.pdf: This is the documentation file. This file contains answers to Question 1 and Question 2. I have inserted the image of the Self Organizing Map in the documentation and have also provided in teh folder with the name "sc_map.png". This answers Question 2(e). For Question 2(f), I have provided the source code with name "SOM_self_computing4.py" inside Question2_SourceCode folder.



