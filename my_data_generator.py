import torchvision.transforms as transforms
from PIL import Image
import os.path
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from tsne import TSNEGen

# constants
config = None
meanRange = None
clusterCoef = None
noClusters = None

# samples a distribution and adds anomalies
def generateDistribution(mean, cov, dataset_size, img_size, clusters, anomalyRatio, use_train_anomalies = True, meanRange = None, clusterCoef = None, noClusters = None):
    mean = (meanRange[0] - meanRange[1]) * mean.double() + meanRange[1]
    cov = cov.double()
    np.random.seed(42)
    #torch.manual_seed(46) #45 e ala bun
    mvn = torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance_matrix=cov)
    
    # sample the normal distribution and reshape each example into a matrix
    samples = mvn.sample((dataset_size,)).float()
    if img_size != 2:
        samples = samples.view(dataset_size, img_size, img_size)
    else:
        samples = samples.view(dataset_size, img_size)

    # calculate no of anomalies and randomly pick the indices from the dataset
    noAnomalies = int(anomalyRatio * samples.shape[0])
    anomalyIndices = np.random.choice(samples.shape[0], size=noAnomalies, replace=False)
    labels = torch.zeros(samples.shape[0])
    # for each anomaly randomly pick a cluster from the anomaly array and place it into the cluster
    if noClusters > 0 and use_train_anomalies == True:
        for index in anomalyIndices:
            cluster = clusters[np.random.randint(0, clusters.shape[0])]
            samples[index] += clusterCoef * cluster
        labels[anomalyIndices] += 1
    return samples, labels

# generate a valid covariance matrix of size n x n 
def generateCovarianceMatrix(n, m = None):
    validMatrix = False
    if m == None:
        m = n
    np.random.seed(42)
    # keep generating random matrices until a valid covariance matrix is obtained
    while not validMatrix:
        randMatrix =  np.random.rand(n, m)
        
        covMatrix = np.dot(randMatrix, randMatrix.T)

        eigvals = np.linalg.eigvals(covMatrix)
        if np.all(eigvals > 0):
            validMatrix = True
    return torch.tensor(covMatrix)

# generate k random matrices of size n x m to represent a new cluster
def generateClusters(n, m, k):
    clusters = torch.tensor([])
    if m == 0:
        for i in range(k):
            clusters = torch.cat((clusters, -1 * torch.randn(1, n) + 1), dim=0)
    else:
        for i in range(k):
            clusters = torch.cat((clusters, -1 * torch.randn(1, n, m) + 1), dim=0)
    return clusters

def rotate_covariance_matrix(cov_matrix, rotation_degrees, num_iterations):
    # Convert the covariance matrix to a NumPy array
    cov_matrix = np.asarray(cov_matrix)

    # Compute the eigendecomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Convert the rotation angles to radians
    rotation_radians = np.radians(rotation_degrees)

    # Calculate the step size for interpolation
    step_size = rotation_radians / num_iterations

    rotated_matrices = []

    # Iterate over the desired number of iterations
    for i in range(num_iterations):
        # Calculate the rotation angle for the current iteration
        current_rotation = step_size * (i + 1)

        # Compute the intermediate rotation matrix
        rotation_matrix = np.eye(cov_matrix.shape[0])
        rotation_matrix[0, 0] = np.cos(current_rotation)
        rotation_matrix[0, 1] = -np.sin(current_rotation)
        rotation_matrix[1, 0] = np.sin(current_rotation)
        rotation_matrix[1, 1] = np.cos(current_rotation)
        # Rotate the eigenvectors by the intermediate rotation matrix
        rotated_eigenvectors = eigenvectors @ rotation_matrix

        # Recreate the intermediate covariance matrix
        rotated_cov_matrix = rotated_eigenvectors @ np.diag(eigenvalues) @ rotated_eigenvectors.T

        eigenvalues, eigenvectors = np.linalg.eigh(rotated_cov_matrix)

        rotation_matrix = np.eye(cov_matrix.shape[0])
        rotation_matrix[-1, -1] = np.cos(current_rotation)
        rotation_matrix[-1, -2] = -np.sin(current_rotation)
        rotation_matrix[-2, -1] = np.sin(current_rotation)
        rotation_matrix[-2, -2] = np.cos(current_rotation)
        # Rotate the eigenvectors by the intermediate rotation matrix
        rotated_eigenvectors = eigenvectors @ rotation_matrix

        # Recreate the intermediate covariance matrix
        rotated_cov_matrix = rotated_eigenvectors @ np.diag(eigenvalues) @ rotated_eigenvectors.T

        eigenvalues, eigenvectors = np.linalg.eigh(rotated_cov_matrix)

        rotation_matrix = np.eye(cov_matrix.shape[0])
        rotation_matrix[0, 0] = np.cos(current_rotation)
        rotation_matrix[-1, 0] = -np.sin(current_rotation)
        rotation_matrix[0, -1] = np.sin(current_rotation)
        rotation_matrix[-1, -1] = np.cos(current_rotation)
        # Rotate the eigenvectors by the intermediate rotation matrix
        rotated_eigenvectors = eigenvectors @ rotation_matrix

        # Recreate the intermediate covariance matrix
        rotated_cov_matrix = rotated_eigenvectors @ np.diag(eigenvalues) @ rotated_eigenvectors.T

        rotated_matrices.append(torch.tensor(rotated_cov_matrix))

    return rotated_matrices

def calculate_average(left, right, increments):
    if increments == 0:
        return []

    average = (left + right) / 2

    left_result = calculate_average(left, average, increments - 1)
    right_result = calculate_average(average, right, increments - 1)

    result = left_result + [average] + right_result

    return result


def generateData(ratio, dimensions = 0, dataset_size = 0, use_train_anomalies = None, config = None):

    meanRange = (0, config['mean_max'][0])
    clusterCoef = config['clusterCoef'][0]
    noClusters = config['no_clusters'][0]

    anomalyRatio = ratio
    if dimensions == 2:
        mean = torch.rand(dimensions)
        clusters = generateClusters(1, 1, noClusters)
        cov = generateCovarianceMatrix(dimensions)
    else:
        mean = torch.rand(dimensions ** 2)
        clusters = generateClusters(dimensions, dimensions, noClusters)
        cov = generateCovarianceMatrix(dimensions ** 2)   
    # if special == 'oval':
    #     cov = generateCovarianceMatrix(dimensions**2, special = 'oval')
    anomalyRatio = ratio

    data, labels = generateDistribution(mean, cov, dataset_size, dimensions, clusters, anomalyRatio, use_train_anomalies, meanRange, clusterCoef, noClusters)
    # tsne = TSNEGen(data, labels)
    # tsne.generateTSNE()

    # sns.jointplot(x=data[:, 0], y=data[:, 1], hue=labels, kind='scatter')

    # # Display the plot
    # plt.show()
    return data, labels

def generateDataRotated(ratio, dimensions = 0, dataset_size = 0, use_train_anomalies = None, config = None):
    rotation = config['rotation'][0]
    train_classes = config['no_classes'][0]
    test_classes = config['no_classes_test'][0]
    increments = config['no_classes'][0] + config['no_classes_test'][0]

    meanRange = (0, config['mean_max'][0])
    clusterCoef = config['clusterCoef'][0]
    noClusters = config['no_clusters'][0]

    x_train = []
    y_train = []
    y_test = []
    x_test = []
    if dimensions == 2:
        mean = torch.rand(dimensions)
        clusters = generateClusters(1, 1, noClusters)
        cov = generateCovarianceMatrix(dimensions)
    else:
        mean = torch.rand(dimensions ** 2)
        clusters = generateClusters(dimensions, dimensions, noClusters)
        cov = generateCovarianceMatrix(dimensions ** 2)        
    covMatrices = rotate_covariance_matrix(cov, rotation, increments)
    for i in range(train_classes):
        cov1 = covMatrices[i]
        data, labels = generateDistribution(mean, cov1, dataset_size, dimensions, clusters, ratio, use_train_anomalies, meanRange, clusterCoef, noClusters)

        if dimensions != 2:
            x = data.reshape((dataset_size, dimensions ** 2))
        else:
            x = data
        x_train.append(x)
        y_train.append(labels)
    
    for i in range(train_classes, increments):
        cov1 = covMatrices[i]
        data, labels = generateDistribution(mean, cov1, dataset_size, dimensions, clusters, ratio, use_train_anomalies, meanRange, clusterCoef, noClusters)

        x = data.reshape((dataset_size, dimensions * dimensions))
        x_test.append(x)
        y_test.append(labels)

    return x_train, x_test, y_train, y_test

def generateDataAvg(ratio, dimensions = 0, dataset_size = 0, use_train_anomalies = None, config = None):
    rotation = config['rotation'][0]
    train_classes = config['no_classes'][0]
    test_classes = config['no_classes_test'][0]
    increments = config['no_classes'][0] + config['no_classes_test'][0]

    meanRange = (0, config['mean_max'][0])
    clusterCoef = config['clusterCoef'][0]
    noClusters = config['no_clusters'][0]

    x_train = []
    y_train = []
    y_test = []
    x_test = []

    mean = torch.rand(dimensions**2)
    clusters = generateClusters(dimensions, dimensions, noClusters)
    cov1 = generateCovarianceMatrix(dimensions**2)
    cov2 = generateCovarianceMatrix(dimensions**2)
    covMatrices = calculate_average(cov1, cov2, 5)
    totalClasses = len(covMatrices)
    train_classes = int(totalClasses * 0.8)
    test_classes = totalClasses - train_classes
    print(covMatrices[0])
    print(covMatrices[15])
    print(covMatrices[-1])
    for i in range(train_classes):
        cov1 = covMatrices[i]
        data, labels = generateDistribution(mean, cov1, dataset_size, dimensions, clusters, ratio, use_train_anomalies, meanRange, clusterCoef, noClusters)

        x = data.reshape((dataset_size, dimensions ** 2))

        # tsne = TSNEGen(x, labels)
        # tsne.generateTSNE()

        x_train.append(x)
        y_train.append(labels)
    
    for i in range(train_classes, totalClasses):
        cov1 = covMatrices[i]
        data, labels = generateDistribution(mean, cov1, dataset_size, dimensions, clusters, ratio, use_train_anomalies, meanRange, clusterCoef, noClusters)

        x = data.reshape((dataset_size, dimensions * dimensions))
        x_test.append(x)
        y_test.append(labels)

    return x_train, x_test, y_train, y_test
