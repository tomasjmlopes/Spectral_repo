# import numpy as np
import cupy as cp

class GPUKMeans:
    def __init__(
        self,
        n_clusters,
        distance='euclidean',
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=None,
        verbose=True
    ):
        self.n_clusters = n_clusters
        self.distance = distance
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self._set_distance_function()

    def _set_distance_function(self):
        distance_functions = {
            'euclidean': self._euclidean_distance,
            'hamming': self._hamming_distance,
            'correlation': self._correlation_distance
        }
        self._distance_func = distance_functions.get(self.distance)
        if self._distance_func is None:
            raise ValueError(f"Unsupported distance metric: {self.distance}")

    def _euclidean_distance(self, X, Y):
        return cp.sqrt(cp.sum((X[:, cp.newaxis] - Y) ** 2, axis=2))

    def _hamming_distance(self, X, Y):
        return cp.mean(X[:, cp.newaxis] != Y, axis=2)

    def _correlation_distance(self, X, Y):
        X_centered = X - cp.mean(X, axis=1, keepdims=True)
        Y_centered = Y - cp.mean(Y, axis=1, keepdims=True)
        X_norm = cp.linalg.norm(X_centered, axis=1, keepdims=True)
        Y_norm = cp.linalg.norm(Y_centered, axis=1, keepdims=True)
        corr = cp.dot(X_centered, Y_centered.T) / (X_norm * Y_norm.T)
        return 1 - corr

    def _init_centroids(self, X):
        n_samples = X.shape[0]
        centroids = cp.empty((self.n_clusters, X.shape[1]), dtype=X.dtype)
        
        # Choose the first centroid randomly
        centroids[0] = X[cp.random.choice(n_samples, size = 1)]
        
        # Compute the remaining k-1 centroids
        for k in range(1, self.n_clusters):
            distances = cp.min(self._distance_func(X, centroids[:k]), axis=1)
            probabilities = distances / cp.sum(distances)
            cumulative_probabilities = cp.cumsum(probabilities)
            r = cp.random.rand()
            next_centroid_index = cp.searchsorted(cumulative_probabilities, r)
            centroids[k] = X[next_centroid_index]
        
        return centroids

    def _compute_inertia(self, X, labels, centroids):
        distances = self._distance_func(X, centroids)
        return cp.sum(distances[cp.arange(len(X)), labels] ** 2)

    def fit(self, X):
        X = cp.asarray(X, dtype=cp.float32)
        best_inertia = float('inf')
        best_labels = None
        best_centroids = None

        for _ in range(self.n_init):
            labels, centroids, inertia = self._kmeans_single(X)
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centroids = centroids

        self.labels_ = best_labels
        self.cluster_centers_ = best_centroids
        self.inertia_ = best_inertia
        return self

    def _kmeans_single(self, X):
        centroids = self._init_centroids(X)
        
        for iteration in range(self.max_iter):
            old_centroids = centroids.copy()

            # Assign samples to closest centroids
            distances = self._distance_func(X, centroids)
            labels = cp.argmin(distances, axis=1)

            # Update centroids
            for i in range(self.n_clusters):
                mask = (labels == i)
                if cp.any(mask):
                    centroids[i] = cp.mean(X[mask], axis=0)
                else:
                    centroids[i] = X[cp.random.randint(X.shape[0])]

            # Check for convergence
            centroid_shift = cp.linalg.norm(centroids - old_centroids)
            if self.verbose:
                print(f"Iteration {iteration+1}, Centroid Shift: {centroid_shift:.6f}", end='\r')
            if centroid_shift < self.tol:
                break

        if self.verbose:
            print()  # For a new line after the loop completes
        inertia = self._compute_inertia(X, labels, centroids)
        return labels, centroids, inertia

    def predict(self, X):
        X = cp.asarray(X, dtype=cp.float32)
        distances = self._distance_func(X, self.cluster_centers_)
        return cp.argmin(distances, axis=1)

    def fit_predict(self, X):
        return self.fit(X).predict(X)

    def get_labels(self):
        return cp.asnumpy(self.labels_)

    def get_cluster_centers(self):
        return cp.asnumpy(self.cluster_centers_)

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    # Generate sample data
    X, _ = make_blobs(n_samples=1000, centers=5, cluster_std=1.0, random_state=42)

    # Initialize and fit KMeans
    kmeans = GPUKMeans(n_clusters=5, distance='euclidean')
    kmeans.fit(X)

    # Output results
    print(f'Inertia: {kmeans.inertia_}')
    print(f'Centroids:\n{kmeans.get_cluster_centers()}')
    print(f'Labels:\n{kmeans.get_labels()}')


# import cupy as cp
# from tqdm import tqdm

# class GPUKMeans:
#     def __init__(
#         self,
#         n_clusters,
#         distance='euclidean',
#         n_init=10,
#         max_iter=300,
#         tol=1e-4,
#         random_state=None,
#         verbose=True
#     ):
#         self.n_clusters = n_clusters
#         self.distance = distance
#         self.n_init = n_init
#         self.max_iter = max_iter
#         self.tol = tol
#         self.random_state = random_state
#         self.verbose = verbose
#         self.cluster_centers_ = None
#         self.labels_ = None
#         self.inertia_ = None
#         self._set_distance_function()

#     def _set_distance_function(self):
#         distance_functions = {
#             'euclidean': self._euclidean_distance,
#             'hamming': self._hamming_distance,
#             'correlation': self._correlation_distance
#         }
#         self._distance_func = distance_functions.get(self.distance)
#         if self._distance_func is None:
#             raise ValueError(f"Unsupported distance metric: {self.distance}")

#     def _euclidean_distance(self, X, Y):
#         return cp.sum((X[:, cp.newaxis] - Y) ** 2, axis=2)

#     def _hamming_distance(self, X, Y):
#         return cp.mean(X[:, cp.newaxis] != Y, axis=2)

#     def _correlation_distance(self, X, Y):
#         X_centered = X - cp.mean(X, axis=1, keepdims=True)
#         Y_centered = Y - cp.mean(Y, axis=1, keepdims=True)
#         X_norm = cp.linalg.norm(X_centered, axis=1, keepdims=True)
#         Y_norm = cp.linalg.norm(Y_centered, axis=1, keepdims=True)
#         corr = cp.dot(X_centered, Y_centered.T) / (X_norm * Y_norm.T)
#         return 1 - corr

#     def _init_centroids(self, X):
#         n_samples = X.shape[0]
#         indices = cp.random.choice(n_samples, self.n_clusters, replace=False)
#         return X[indices]

#     def _compute_inertia(self, X, labels, centroids):
#         distances = self._distance_func(X, centroids)
#         return cp.sum(distances[cp.arange(len(X)), labels])

#     def fit(self, X):
#         X = cp.asarray(X, dtype=cp.float32)
#         best_inertia = float('inf')
#         best_labels = None
#         best_centroids = None

#         init_pbar = tqdm(range(self.n_init), desc="Initializations", disable=not self.verbose)
#         for _ in init_pbar:
#             labels, centroids, inertia = self._kmeans_single(X)
#             if inertia < best_inertia:
#                 best_inertia = inertia
#                 best_labels = labels
#                 best_centroids = centroids
#             init_pbar.set_postfix({"Best inertia": f"{best_inertia:.4f}"})

#         self.labels_ = best_labels
#         self.cluster_centers_ = best_centroids
#         self.inertia_ = best_inertia
#         return self

#     def _kmeans_single(self, X):
#         centroids = self._init_centroids(X)
        
#         iter_pbar = tqdm(range(self.max_iter), desc="K-means iterations", disable=not self.verbose)
#         for iteration in iter_pbar:
#             old_centroids = centroids.copy()

#             # Assign samples to closest centroids
#             distances = self._distance_func(X, centroids)
#             labels = cp.argmin(distances, axis=1)

#             # Update centroids
#             for i in range(self.n_clusters):
#                 mask = (labels == i)
#                 if cp.any(mask):
#                     centroids[i] = cp.mean(X[mask], axis=0)
#                 else:
#                     centroids[i] = X[cp.random.randint(X.shape[0])]

#             # Check for convergence
#             centroid_shift = cp.linalg.norm(centroids - old_centroids)
#             iter_pbar.set_postfix({"centroid_shift": f"{centroid_shift:.6f}"})
#             if centroid_shift < self.tol:
#                 break

#         inertia = self._compute_inertia(X, labels, centroids)
#         return labels, centroids, inertia

#     def predict(self, X):
#         X = cp.asarray(X, dtype=cp.float32)
#         distances = self._distance_func(X, self.cluster_centers_)
#         return cp.argmin(distances, axis=1)

#     def fit_predict(self, X):
#         return self.fit(X).predict(X)

#     def get_labels(self):
#         return cp.asnumpy(self.labels_)

#     def get_cluster_centers(self):
#         return cp.asnumpy(self.cluster_centers_)