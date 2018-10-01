import os
import matplotlib.pyplot as plt
import click
import random
import math
import numpy as np
from PIL import Image # module Pillow

def generate_clusters(k = 4, size = 100):

    # random.seed(100)

    clusters = []

    click.secho('Generating {} clusters with {} samples in each.'
    .format(k, size), fg='green')

    for _ in range(k):
        meanx = random.uniform(-10, 10)
        meany = random.uniform(-10, 10)
        std = random.uniform(0.5, 1)

        click.secho('Cluster {}: means = ({}, {}) and std = {}'
        .format(k, meanx, meany, std))

        for _ in range(size):
            x = random.gauss(meanx, std)
            y = random.gauss(meany, std)
            clusters.append((x, y))
        
    clusters = list(zip(*list(clusters)))
    
    return clusters

def calculate_centroid(samples, membership, cluster):

    # Weighted average calculation
    sum_u = sum(membership[:, cluster] ** 2)
    dimensions = []
    for dimension in range(len(samples[0])):
        dimensions.append(sum([samples[index][dimension] * membership[index, cluster] ** 2 for index in range(len(samples))]) / sum_u)
        # centroid_y = sum([samples[index][1] * membership[index, cluster] ** 2 for index in range(len(samples))]) / sum_u

    # Numpy weighted average calculation, supose to be faster than the method above
    # centroid_x = np.average([sample[0] for sample in samples], weights=membership[:, 0] ** 2)
    # centroid_y = np.average([sample[1] for sample in samples], weights=membership[:, 0] ** 2)

    return tuple(dimensions) 

def euclidian(a, b):
    dist = 0
    for dim in range(len(a)):
        dist += (a[dim] - b[dim]) ** 2
    
    return math.sqrt(dist)

def update_membership(samples, centroids):

    # Initializing membership matrix
    uMatrix = []

    newMembership = np.zeros([len(samples), len(centroids)])

    # Anonymal function to calculate new membership value
    update = lambda dist, dists: 1 / (math.pow((sum([dist / sumu for sumu in dists]) ), 2))

    # Generate membership matrix. The columns are the clusters, and each line is a sample
    for i in range(len(samples)):
        dists = []
        for centroid in centroids.keys():
            dists += [euclidian(samples[i], centroids.get(centroid))]

        newMembership[i][:] = [update(distU, dists) for distU in dists]

    return newMembership

def cmeans(samples, k):

    # Generate initial membership
    click.echo('Generate initial membership')
    labels = []
    for _ in range(len(samples[0] * k)):
        labels.append(random.uniform(0,(1/k)))
        # labels.append(1 / k)
    
    oldMembership = np.reshape(labels, [len(samples[0]), k])

    # Start main classification loop
    click.echo('Start main classification loop')
    change_membership = 0
    generation = 0
    while(change_membership < 3 and generation <= 1):

        generation += 1

        # Calculate the centroids
        centroids = dict()
        for cluster in range(k):
            k_samples = list(zip(*samples))
            centroids[cluster] = calculate_centroid(k_samples, oldMembership, cluster)
        
        # Update membership matrix
        newMembership = update_membership(k_samples, centroids)
        click.secho('[Gen {}] {} '.format(generation, centroids), fg='yellow')

        oldLabel = np.argmax(oldMembership, axis=1)
        newLabel = np.argmax(newMembership, axis=1)

        if np.array_equal(newLabel, oldLabel):
            change_membership += 1

        oldMembership = newMembership
    
    click.secho('Finished main classification loop in {} generations.'
                 .format(generation), fg='green')

    samples.append(newLabel)

    return [list(samples), centroids]

def main(k = 4):
    click.clear()
    samples = generate_clusters(k = k)

    samples, centroids = cmeans(samples, k)
    centroids = list(zip(*list(centroids.values())))


    plt.scatter(samples[0], samples[1], c=samples[2])
    plt.scatter(centroids[0], centroids[1], marker='*', c='red')
    plt.title('Samples Classificated and Centroids Positions')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    

def cluster_images():
    click.clear()

    dire = os.path.join('ImagensTeste')
    images_in = os.listdir(dire)
    # for image in images_in:
    image = images_in[0]
    img = Image.open(os.path.join(dire, image))
    width, height = img.size
    pixels = list(zip(*list(img.getdata())))

    pixels, centroids = cmeans(pixels, k=3)

    pixels = list(zip(list(pixels)))

    click.secho('Generating new image file')
    newPixels = [centroids.get(pixel[-1]) for pixel in pixels]
    newPixels = np.array(newPixels)
    newImage = Image.fromarray(newPixels.reshape(width, height))
    newImage.save(image + '_converted', 'JPEG')

    click.secho('Finished alghorithm.', fg='green')







if __name__ == '__main__':
    # main()
    cluster_images()
