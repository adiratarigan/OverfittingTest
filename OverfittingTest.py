#Adira Kurniawan 15/378052/PA/16527


import seaborn
import matplotlib.pyplot as plotting




def hitung_h(x1,x2,x3,x4,theta1,theta2,theta3,theta4,bias,i):
    h = x1[i]*theta1+x2[i]*theta2+x3[i]*theta3+x4[i]*theta4+bias
    return h

def aktivasi(h):
    
    e = 2.71828
    activation = 1/(1+e**(-h))

    return activation

def hitung_pred(s):
    if s<0.5:
        pred = 0
    else:
        pred = 1
    return pred

def preprocess(input):
    output_array = []
    for item in input:
        output_array.append(float(item))
    return output_array

def split_train_test(x1,x2,x3,x4,label):

    vx1 = x1[40:50] + x1[90:100]
    vx2 = x2[40:50] + x2[90:100]
    vx3 = x3[40:50] + x3[90:100]
    vx4 = x4[40:50] + x4[90:100]
    label_val = label[40:50] + label[50:90]
    x1 = x1[0:40] + x1[50:90]
    x2 = x2[0:40] + x2[50:90]
    x3 = x3[0:40] + x3[50:90]
    x4 = x4[0:40] + x4[50:90]
    label = label[0:40] + label[50:90]

    return x1,x2,x3,x4,label,vx1,vx2,vx3,vx4,label_val

def calculate_delta_theta(activation,pred,label,x1,x2,x3,x4,i):

    dbias = 2*(activation-label[i])*(1-activation)*activation
    dtheta1 = x1[i]*dbias
    dtheta2 = x2[i]*dbias
    dtheta3 = x3[i]*dbias
    dtheta4 = x4[i]*dbias

    return dtheta1,dtheta2,dtheta3,dtheta4,dbias

def update_theta(dtheta1,dtheta2,dtheta3,dtheta4,dbias,theta1,theta2,theta3,theta4,bias,alpha):

    theta1 = theta1-dtheta1*alpha
    theta2 = theta2-dtheta2*alpha
    theta3 = theta3-dtheta3*alpha
    theta4 = theta4-dtheta4*alpha
    bias = bias-dbias*alpha

    return theta1,theta2,theta3,theta4,bias

def calculate_error(label,activation,i,error_array):

    error = (label[i]-activation)**2
    error_array[i]=error

    return error
 
def average_error_per_epoch(average_error_array,error_array):

    avg_error = sum(error_array) / float(len(error_array))
    average_error_array.append(avg_error)

    return avg_error

def predict(x1,x2,x3,x4,theta1,theta2,theta3,theta4,bias):

    h = x1*theta1+x2*theta2+x3*theta3+x4*theta4+bias
    s = aktivasi(h)

    if s<0.5:
        pred = 0
        notes = "Iris - Setosa"
    else:
        pred = 1
        notes = "Iris - Versicolor"
    
    return pred, notes, s

def validation(vx1,vx2,vx3,vx4,label_val,theta1,theta2,theta3,theta4,bias,validation_array):

    activation_array = []
    tmp = []

    for i in range(len(vx1)):
        _, _, s = predict(vx1[i],vx2[i],vx3[i],vx4[i],theta1,theta2,theta3,theta4,bias)
        activation_array.append(s)

    for i in range(len(activation_array)):
        error = (label_val[i]-activation_array[i])**2
        tmp.append(error)

    average_error_per_epoch(validation_array,tmp)

    return 0
    
def plot(validation_array,average_error_array):

    fig = plotting.figure()
    fig.suptitle('Grafik MSE')
    seaborn.set_style("white")

    plotting.plot(list(range(1,61)),validation_array, label = "Validasi")
    plotting.plot(list(range(1,61)),average_error_array, label = "Training")

    plotting.xlabel('Epoch')
    plotting.ylabel('MSE')
    plotting.legend()

    return plotting.show()

def main():

        # Data input
    with open("iris.txt", "r") as data:
        data = data.read().replace('\n', ',')
        x1 = preprocess(data.split(",")[0::5])
        x2 = preprocess(data.split(",")[1::5])
        x3 = preprocess(data.split(",")[2::5])
        x4 = preprocess(data.split(",")[3::5])
        label = preprocess(data.split(",")[4::5])
    x1,x2,x3,x4,label,vx1,vx2,vx3,vx4,label_val = split_train_test(x1,x2,x3,x4,label)

  
    theta1 = 0.2
    theta2 = 0.2
    theta3 = 0.2
    theta4 = 0.2
    bias = 0.3
    
    
    print("\nPilihan Learning rate, \n1 = 0.1 \n2 = 0.8")
    uinput = float(input("Pilih learning rate: "))
    
       
    if (uinput == 1) :
        alpha = 0.1
    elif uinput == 2:
        alpha = 0.8
        
    
    training_epoch = 60
    error_array_per_epoch=[None]*80 
    average_error_array=[]
    validation_array=[]

   
    avg_error = 0.0
    prev = 0.0
    tmp = True

    print("\n")
    
    for x in range(training_epoch):

  
        for i in range(len(label)):
            h = hitung_h(x1,x2,x3,x4,theta1,theta2,theta3,theta4,bias,i)
            activation = aktivasi(h)
            pred = hitung_pred(activation)
            error = calculate_error(label,activation,i,error_array_per_epoch)
            dtheta1,dtheta2,dtheta3,dtheta4,dbias = calculate_delta_theta(activation,pred,label,x1,x2,x3,x4,i)
            theta1,theta2,theta3,theta4,bias = update_theta(dtheta1,dtheta2,dtheta3,dtheta4,dbias,theta1,theta2,theta3,theta4,bias,alpha)

        prev = avg_error
        avg_error = average_error_per_epoch(average_error_array,error_array_per_epoch)
        validation(vx1,vx2,vx3,vx4,label_val,theta1,theta2,theta3,theta4,bias,validation_array)

        if abs(avg_error-prev)<0.0001 and (tmp==True and x>0):
            tmp = False
            print("Jumlah epoch untuk convergence :",x+1)
          
  
    plot(validation_array,average_error_array)
   
main()