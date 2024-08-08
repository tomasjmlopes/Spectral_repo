import numpy as np
import torch
from tqdm import tqdm
from functools import partial

def initialize(X, num_clusters, seed):
    """
    Initialize cluster centers
    """

    num_samples = len(X)
    if seed == None:
        indices = np.random.choice(num_samples, num_clusters, replace = False)
    else:
        np.random.seed(seed)
        indices = np.random.choice(num_samples, num_clusters, replace = False)
    initial_state = X[indices]
    return initial_state


def kmeans(
        X, 
        num_clusters,
        distance = 'euclidean',
        cluster_centers = [],
        tol = 1e-4,
        tqdm_flag = True,
        iter_limit = 0,
        device = torch.device('cpu'),
        seed = None
):
    """
    perform kmeans
    """
    if tqdm_flag:
        print(f'Running Kmeans on {device}..')
    
    if distance == 'euclidean':
        pairwise_distance_function = partial(pairwise_distance, device = device, tqdm_flag = tqdm_flag)
    elif distance == 'iou':
        pairwise_distance_function = partial(pairwise_iou, device = device, tqdm_flag = tqdm_flag)
    else:
        raise NotImplementedError

    # Convert to Float
    X = X.float()

    # Transfer to Device
    X = X.to(device)

    if type(cluster_centers) == list:
        initial_state = initialize(X, num_clusters, seed = seed)
    else:
        if tqdm_flag:
            print('Resuming')

        # Find closest point to initial cluster center
        initial_state = cluster_centers
        dis = pairwise_distance_function(X, initial_state)
        choice_points = torch.argmin(dis, dim = 0)
        initial_state = X[choice_points]
        initial_state = initial_state.to(device)

    iteration = 0
    if tqdm_flag:
        tqdm_meter = tqdm(desc = '[Running Kmeans]')
    while True:

        dis = pairwise_distance_function(X, initial_state)
        choice_clusters = torch.argmin(dis, dim = 1)
        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(choice_clusters == index).squeeze().to(device)
            selected = torch.index_select(X, 0, selected)

            if selected.shape[0] == 0:
                selected = X[torch.randint(len(X), (1,))]

            initial_state[index] = selected.mean(dim = 0)
        
        center_shift = 0
        center_intersect = 0

        if distance == 'euclidean':
            center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre)**2, dim = 1)
            ))
        # if distance == 'iou':
        #     inter = torch.sum((initial_state.int() & initial_state_pre.int()), dim = 1)
        #     union = torch.sum((initial_state.int() | initial_state_pre.int()), dim = 1)

        #     center_intersect = 1 - inter/union

        # Increment iteration
        iteration += 1
        inertia = 0

        # Update tqdm meter
        if tqdm_flag:
            tqdm_meter.set_postfix(
                iteration = f'{iteration}',
                center_shift = f'{center_shift**2:0.6f}',
                tol = f'{tol:0.6f}'
            )
            tqdm_meter.update()

        if distance == 'euclidean' and center_shift ** 2 < tol:
            inertia = 0
            for index in range(num_clusters):
                selected = torch.nonzero(choice_clusters == index).squeeze().to(device)
                selected = torch.index_select(X, 0, selected)
                inertia += torch.sum(((selected - initial_state[index]) ** 2))
            break
        # if distance == 'iou' and center_intersect < tol:
        #     break
        if iter_limit != 0 and iteration >= iter_limit:
            break

    return choice_clusters.cpu(), initial_state.cpu(), inertia


def kmeans_predict(
        X,
        cluster_center,
        distance = 'euclidean',
        device = torch.device('cpu'),
        tqdm_flag = True
):
    # Predict using cluster centers

    if tqdm_flag:
        print(f'Predicting on {device}')

    if distance == 'euclidean':
        pairwise_distance_function = partial(pairwise_distance, device = device, tqdm_flag = tqdm_flag)
    elif distance == 'iou':
        pairwise_distance_function = partial(pairwise_iou, device = device, tqdm_flag = tqdm_flag)
    else:
        raise NotImplementedError
    
    # Convert to Float
    X = X.float()

    # Transfer to Device
    X = X.to(device)

    dis = pairwise_distance_function(X, cluster_center)
    choice_clusters = torch.argmin(dis, dim = 1)

    return choice_clusters.cpu()


def pairwise_distance(data1, data2, device = torch.device('cpu'), tqdm_flag = True):
    # if tqdm_flag:
    #     print(f'device is :{device}')
    
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim = 1)

    # 1*N*M
    B = data2.unsqueeze(dim = 0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim = -1).squeeze()
    return dis

def pairwise_iou(data1, data2, device = torch.device('cpu'), tqdm_flag = True):
    # if tqdm_flag:
    #     print(f'device is :{device}')
    
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    A = data1.int()
    B = data2.int()

        # N*1*M
    A = A.unsqueeze(dim = 1)

    # 1*N*M
    B = B.unsqueeze(dim = 0)

    inter = torch.sum((A & B), dim = 2)
    inter2 = torch.sum((torch.logical_not(A) & torch.logical_not(B)), dim = 2)
    union = torch.sum((A | B), dim = 2)
    union2 = torch.sum((torch.logical_not(A) | torch.logical_not(B)), dim = 2)

    #inter3 = torch.sum((torch.logical_not(A) & B), dim = 2)
    #union3 = torch.sum((torch.logical_not(A) | B), dim = 2)
    
    dis = 1 - (inter/union )#* inter2/union2))
    # return N*N matrix for pairwise distance
    dis = dis.squeeze()
    return dis