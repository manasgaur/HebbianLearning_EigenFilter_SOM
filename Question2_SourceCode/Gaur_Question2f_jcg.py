import numpy as np

''' reading the input file'''

inputfile = open('animals_from_jcg.dat', 'r')
features = int(inputfile.readline())

animals_features = []

for line in inputfile:
    tokens = line.split()
    animals_features.append(tokens)

# reading the cod file given in the DATA folder

outputfile = open('animals_from_jcg.cod', 'r')
row_read = outputfile.readline()

animals_output_read = []

for line in outputfile:
    tokens = line.split()
    animals_output_read.append(tokens)

outputfile.close()
inputfile.close()

print animals_output_read

animals_output_infloat=np.asarray([[float(y) for y in x[0:features]] for x in animals_output_read])

# labels of each animal
labels = []
for val in animals_output_infloat:
    label = ""
    max = 0
    for inputval in animals_features:
        input_feature_array = np.array(inputval[0:features],dtype=float)
        simval = np.dot(input_feature_array, val)
        if simval > max:
            label = inputval[features]
            max = simval
    #print(label)
    labels.append(label)

# Generating the labeled cod file
count = 0
with open('animals_from_jcg.cod', 'r') as outputfile:
 with open('Gaur_animals_map_labled.cod','w') as lbloutput:
     rdline = outputfile.readline()
     lbloutput.write(rdline)
     for line in outputfile:
        line2 = line.rstrip('\n')+" "+labels[count]+'\n'
        #print(newline)
        lbloutput.write(line2)
        count = count + 1
outputfile.close()