import jax
import jax.numpy as jnp
from sklearn import metrics
from functools import partial

@jax.jit
def compute_consensus_matrix(partitions):
    n_samples = partitions.shape[1]
    x, y = jnp.meshgrid(jnp.arange(n_samples), jnp.arange(n_samples))
    zz = jnp.c_[x.ravel(), y.ravel()]

    def compress(i,j):
        return (partitions[:,i]==partitions[:,j]).sum()

    func = jax.vmap(compress)
    return func(zz[:,0], zz[:,1]).reshape((n_samples, n_samples))/len(partitions)

@jax.jit
def partition_connectivity(y_pred):
    n = len(y_pred)
    x, y = jnp.meshgrid(jnp.arange(n), jnp.arange(n))
    zz = jnp.c_[x.ravel(), y.ravel()]

    def compress(i,j):
        return y_pred[i]==y_pred[j]

    func = jax.vmap(compress)
    y_pred_connectivity = func(zz[:,0], zz[:,1]).reshape((n, n))
    return y_pred_connectivity


@jax.jit
def binary_total_variation(P,Q):
    return jnp.abs(P-Q)

@jax.jit
@partial(jax.vmap, in_axes=[0,None])
def compute_tv_ranking(y_pred, centroid):
    y_pred_connectivity = partition_connectivity(y_pred)
    return binary_total_variation(y_pred_connectivity, centroid).mean()


@jax.jit
def binary_hellinger(P,Q):
    term1 = jnp.square(jnp.sqrt(P)-jnp.sqrt(Q))
    term2 = jnp.square(jnp.sqrt(1-P)-jnp.sqrt(1-Q))
    return (term1+term2)/2

@jax.jit
@partial(jax.vmap, in_axes=[0,None])
def compute_hellinger_ranking(y_pred, centroid):
    y_pred_connectivity = partition_connectivity(y_pred)
    return binary_hellinger(y_pred_connectivity, centroid).mean()


@jax.jit
def binary_kl(P,Q):
    # P can either contain 0 or 1
    # Q can be in [0,1]
    # Possible cases are
    # KL(0|0) => 0
    # KL(1|1) => 0
    # KL(0|a) => -log(1-a); and a cannot be equal to 1
    # KL(1|a) => -log(a); and a cannot be equal to 0
    # Note that KL(0|1) and KL(1|0) are impossible because the second number is an average
    return -jnp.log(1-Q+P*(2*Q-1))

@jax.jit
@partial(jax.vmap, in_axes=[0,None])
def compute_kl_ranking(y_pred, centroid):
    y_pred_connectivity = partition_connectivity(y_pred)
    kl = binary_kl(y_pred_connectivity, centroid)
    # Due to hard clustering, we may evaluate KL(0|0) or KL(1|1) when all models agree
    # For such case, the distance is set to 0 instead of nan/inf
    # Note that KL(0|1) and KL(1|0) are impossible because the second number is an average
    return kl.mean()


def pairwise_score(partitions, method="ari"):
    scores = [0]*len(partitions)

    for i, partition in enumerate(partitions):
        for j, other_partition in enumerate(partitions[i+1:]):
            if method == "ari":
                score = metrics.adjusted_rand_score(partition, other_partition)
            else:
                score = metrics.normalized_mutual_info_score(partition, other_partition)
            scores[i] += score
            scores[i+1+j] += score

    return jnp.array(scores)/(len(scores)-1)