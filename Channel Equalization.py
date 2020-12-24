import numpy as np

h0 = -1
h1 = -1
standard_deviation = -1
mean = 0.0

mu = []
count = []
covariance = []
for i in range(8):
    mu.append([0, 0])
    count.append([0, 0])
    covariance.append([[0, 0], [0, 0]])

def noise():
    s = np.random.normal(mean, standard_deviation , 1)
    return s[0]

def funcN(Ik, Ik_1):
    return h0 * Ik + h1 * Ik_1 + noise()

def multivariate(X, mean_vector, covariance_matrix):
    return (2*np.pi)**(-len(X)/2)*np.linalg.det(covariance_matrix)**(-1/2)*np.exp(-np.dot(np.dot((X-mean_vector).T, np.linalg.inv(covariance_matrix)), (X-mean_vector))/2)

def findPrevOmega(j):
    if (j < 4):
        prev_omega1 = 2 * j
        prev_omega2 = 2 * j + 1
    else:
        prev_omega1 = (j - 4) * 2
        prev_omega2 = (j - 4) * 2 + 1
    return prev_omega1,prev_omega2

def makePrediction(D,length,prev_omega):
    prediction = ""
    last_omega = D[-1].index(max(D[-1]))
    if last_omega < 4:
        prediction = "0" + prediction
    else:
        prediction = "1" + prediction

    i = length - 1
    while (i > 0):
        if prev_omega[i][last_omega] < 4:
            prediction = "0" + prediction
        else:
            prediction = "1" + prediction
        last_omega = prev_omega[i][last_omega]
        i -= 1
    return prediction

def train(trainfile):
    data = trainfile.read()

    for i in range(len(data) - 2):
        Ik = int(data[i + 2])
        Ik_1 = int(data[i + 1])
        Ik_2 = int(data[i])

        xk = funcN(Ik, Ik_1)
        xk_1 = funcN(Ik_1, Ik_2)

        omega = 4*Ik + 2*Ik_1 + Ik_2

        mu[omega][0] += xk
        mu[omega][1] += xk_1

        count[omega][0] += 1
        count[omega][1] += 1

    for i in range(8):
        mu[i][0] /= count[i][0]
        mu[i][1] /= count[i][1]

    #print(mu)

    for i in range(len(data)-2):
        Ik = int(data[i + 2])
        Ik_1 = int(data[i + 1])
        Ik_2 = int(data[i])

        xk = funcN(Ik, Ik_1)
        xk_1 = funcN(Ik_1, Ik_2)

        omega = 4 * Ik + 2 * Ik_1 + Ik_2
        covariance[omega][0][0] += (xk-mu[omega][0])**2
        covariance[omega][0][1] += (xk-mu[omega][0])*(xk_1-mu[omega][1])
        covariance[omega][1][0] += (xk_1-mu[omega][1])*(xk-mu[omega][0])
        covariance[omega][1][1] += (xk_1-mu[omega][1])**2

    for i in range(8):
        covariance[i][0][0] /= count[i][0]
        covariance[i][0][1] /= count[i][0]
        covariance[i][1][0] /= count[i][0]
        covariance[i][1][1] /= count[i][0]

    #print(covariance)


def test1(testfile,fwrite):
    xk = []
    data = testfile.read()

    xk.append(0)
    for i in range(len(data) - 1):
        Ik = int(data[i + 1])
        Ik_1 = int(data[i])
        Xk = funcN(Ik, Ik_1)
        xk.append(Xk)

    D = []
    prev_omega = []
    for i in range(len(xk)):
        D.append([0,0,0,0,0,0,0,0])
        prev_omega.append([0,0,0,0,0,0,0,0])

    mu_vec = np.array(mu)
    covariance_vec = np.array(covariance)

    for i in range(1, len(data)):
        for j in range(8):
            prev_omega1,prev_omega2 = findPrevOmega(j)

            d_omegaIk_omegaIk_1 = np.log(0.5) + np.log(multivariate([xk[i], xk[i-1]], mu_vec[j], covariance_vec[j]))

            if (D[i][j] < D[i-1][prev_omega1] + d_omegaIk_omegaIk_1):
                D[i][j] = D[i-1][prev_omega1] + d_omegaIk_omegaIk_1
                prev_omega[i][j] = prev_omega1

            if (D[i][j] < D[i-1][prev_omega2] + d_omegaIk_omegaIk_1):
                D[i][j] = D[i-1][prev_omega2] + d_omegaIk_omegaIk_1
                prev_omega[i][j] = prev_omega2

    prediction = makePrediction(D,len(data),prev_omega)
    fwrite.write(prediction)


def test2(testfile, fwrite):
    data = testfile.read()
    predicted_bit = ""
    prediction = "11"

    for i in range(len(data) - 2):

        Ik = int(data[i + 2])
        Ik_1 = int(data[i + 1])
        Ik_2 = int(data[i])

        xk = funcN(Ik, Ik_1)
        xk_1 = funcN(Ik_1, Ik_2)

        dis = 99999
        omega = -1

        for j in range(8):
            if ((mu[j][0] - xk) ** 2 + (mu[j][1] - xk_1) ** 2 < dis):
                dis = (mu[j][0] - xk) ** 2 + (mu[j][1] - xk_1) ** 2
                omega = j
        if (omega < 4):
            predicted_bit = "0"
        else:
            predicted_bit = "1"
        prediction += predicted_bit

    fwrite.write(prediction)


def accuracy(testfile, outputfile):
    testdata = testfile.read()
    outputdata = outputfile.read()

    count = 0
    for i in range(len(testdata)):
        if(testdata[i] == outputdata[i]):
            count += 1

    accuracy = count/len(testdata)*100
    print("accuracy",accuracy,"%")




if __name__ == "__main__":
    filepar = open('parameter.txt', 'r')
    lines = filepar.read()
    x = lines.split()
    h0 = float(x[0])
    h1 = float(x[1])
    standard_deviation = float(x[2])

    ######## train ########
    trainfile = open('train.txt', 'r')
    train(trainfile)
    trainfile.close()

    ######## method 1 ########
    testfile = open('test.txt', 'r')
    fwrite = open('out1.txt', 'w')
    test1(testfile,fwrite)
    fwrite.close()
    testfile.close()

    testfile = open('test.txt', 'r')
    outputfile = open('out1.txt', 'r')
    accuracy(testfile, outputfile)
    outputfile.close()
    testfile.close()


    ######## method 2 ########
    testfile = open('test.txt', 'r')
    fwrite = open('out2.txt', 'w')
    test2(testfile,fwrite)
    fwrite.close()
    testfile.close()

    testfile = open('test.txt', 'r')
    outputfile = open('out2.txt', 'r')
    accuracy(testfile, outputfile)
    outputfile.close()
    testfile.close()




