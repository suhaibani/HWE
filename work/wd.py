import numpy as np
import scipy.optimize as optimize
from numpy.linalg import norm


FILE = 'hwe_300.txt'
data_dic = {}

lines = [line.rstrip('\n') for line in open(FILE)]
for line in lines:
	components = line.split(' ')
	data_dic[components[0]] = np.array(components[1:len(components)-1])

user_input_data = {}
inputs = ['x','y1','y2','y3']
input_list = []
for inp in inputs:

	not_found = True
	while not_found:
		inp_val = str(raw_input('Please enter value for '+ str(inp) + ' : '))
		inp_val = inp_val.strip()

		if inp_val not in data_dic:
			print('not found')
			continue
		else:
			not_found = False
			user_input_data[inp] = data_dic[inp_val]
			input_list.append(inp_val)




x = user_input_data['x']
y1 = user_input_data['y1']
y2 = user_input_data['y2']
y3 = user_input_data['y3']


x = x.astype(np.float)
y1 = y1.astype(np.float)
y2 = y2.astype(np.float)
y3 = y3.astype(np.float)

#x = x/x.sum(axis=0,keepdims=1)
#y1 =  y1 / y1.sum(axis=0, keepdims=1)
#y2 =  y2 / y2.sum(axis=0, keepdims=1)
#y3 =  y3 / y3.sum(axis=0, keepdims=1)

def f(params):

	#params = params.flatten()
	#params =  params / params.sum(axis=0, keepdims=1)
	a, b, c = params 
	#inter_res = norm(np.subtract ( x, np.subtract(  np.dot(a, y1) , np.subtract(np.dot(b,y2), np.dot(c,y3)))))
	inter_res = norm(np.subtract ( x, np.subtract(  a * y1 , np.subtract( b * y2,  c * y3 )))) ** 2

	#inter_res = inter_res / inter_res.sum(axis=0, keepdims=1)
	#return norm(inter_res)
	return inter_res

initial_guess = [1, 1, 1]
initial_guess = np.array(initial_guess)
initial_guess = initial_guess.astype(np.float)

initial_guess =  initial_guess / initial_guess.sum(axis=0, keepdims=1)

result = optimize.minimize(f, initial_guess , method = 'Nelder-Mead')
if result.success:
    fitted_params = result.x


    print('Output = {} and static output with [0.333,0.333,0.333] = {}'.format(   f(fitted_params),  f(np.array([0.333, 0.333, 0.333]))   ))

    file = open("output.txt", "w")
    output_string = '{}, {}, {}, {} = {}, {}, {}'.format(input_list[0], input_list[1], input_list[2], input_list[3], fitted_params[0], fitted_params[1], fitted_params[2])
    file.write(output_string)
    file.close()

    print("Completed Successfully !")
    print(fitted_params)


else:
    raise ValueError(result.message)


