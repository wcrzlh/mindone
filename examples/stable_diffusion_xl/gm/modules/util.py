from mindspore import Tensor, ops
from mindspore.communication import create_group, get_group_size, get_rank


def linear(x, weight, bias):
    x_shape = x.shape
    if len(x_shape) != 2:
        x = x.reshape(-1, x_shape[-1])
    x = ops.MatMul(transpose_b=True)(x, weight)
    x = ops.bias_add(x, bias)
    if len(x_shape) != 2:
        out_shape = x_shape[:-1] + (x.shape[-1],)
        x = x.reshape(*out_shape)
    return x


def normalize(input: Tensor, p: float = 2.0, dim: int = 1, eps: float = 1e-12) -> Tensor:
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.

    For a tensor :attr:`input` of sizes :math:`(n_0, ..., n_{dim}, ..., n_k)`, each
    :math:`n_{dim}` -element vector :math:`v` along dimension :attr:`dim` is transformed as

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}.

    With the default arguments it uses the Euclidean norm over vectors along dimension :math:`1` for normalization.

    Args:
        input: input tensor of any shape
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
        eps (float): small value to avoid division by zero. Default: 1e-12
    """

    denom = input.norm(p, dim, keepdim=True).clip(min=eps).broadcast_to(input.shape)

    return input / denom


"""Ring Attention utils."""

_SEQUENCE_PARALLEL_GROUP = None
_SEQUENCE_PARALLEL_GLOBAL_RANKS = None
_SEQUENCE_PARALLEL_GROUP_INDEX = None


def init_sp_group(
    sp=1,
) -> None:
    """Initialize parallel groups."""
    world_size = get_group_size()
    if not isinstance(sp, int):
        raise TypeError(f"The input of sp must be int, but get the type of {type(sp)}")
    if sp > world_size:
        raise ValueError(
            f"The sp must be smaller or equal to total device_num, but got the sp is {sp},"
            f"the total device_num is {world_size}"
        )
    if sp & (sp - 1) != 0:
        raise ValueError(f"The sp value must be power of two, but got sp is {sp}")

    dp = world_size // sp
    # Build the context-parallel groups.
    global _SEQUENCE_PARALLEL_GROUP
    global _SEQUENCE_PARALLEL_GLOBAL_RANKS
    global _SEQUENCE_PARALLEL_GROUP_INDEX
    # assert _SEQUENCE_PARALLEL_GROUP is None, 'sequence parallel group is already initialized'

    for j in range(dp):
        start_rank = j * sp
        end_rank = (j + 1) * sp
        ranks = list(range(start_rank, end_rank))
        cur_rank = get_rank()
        if cur_rank in ranks:
            sp_group = "_".join(str(n) for n in ranks)
            sp_group_name = "sp_group_" + sp_group
            create_group(sp_group_name, ranks)
            _SEQUENCE_PARALLEL_GROUP = sp_group_name
            _SEQUENCE_PARALLEL_GLOBAL_RANKS = ranks
            _SEQUENCE_PARALLEL_GROUP_INDEX = j


def get_sequence_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    return _SEQUENCE_PARALLEL_GROUP


def get_sequence_parallel_global_ranks():
    """Get all global ranks of the sequence parallel group that the caller rank belongs to."""
    return _SEQUENCE_PARALLEL_GLOBAL_RANKS


def get_sequence_parallel_world_size():
    """Return world size for the sequence parallel group."""
    return get_group_size(group=get_sequence_parallel_group())


def get_sequence_parallel_rank():
    """Return my rank for the sequence parallel group."""
    return get_rank(group=get_sequence_parallel_group())


def get_sequence_parallel_group_index():
    """Get the sequence parallel group index the caller rank belongs to."""
    return _SEQUENCE_PARALLEL_GROUP_INDEX


def get_sp_chuncks(batch, dp, sp, seq_dim=0, batch_dim=1, enable_dp_shard=True):
    """
    Slice batch input along sequence dimension into multiple chunks,
    which are parallelized across NPUs in a sequence parallel group.
    """
    sp_rank = get_sequence_parallel_rank()
    world_size = get_group_size()
    if not isinstance(dp, int):
        raise TypeError(f"The type of dp must be int, but got the {type(dp)}")
    if not isinstance(sp, int):
        raise TypeError(f"The type of sp must be int, but got the {type(sp)}")
    if not isinstance(seq_dim, int):
        raise TypeError(f"The type of seq_dim must be int, but got the {type(seq_dim)}")
    if not isinstance(batch_dim, int):
        raise TypeError(f"The type of batch_dim must be int, but got the {type(batch_dim)}")
    if not isinstance(enable_dp_shard, bool):
        raise TypeError(f"The type of enable_dp_shard must be bool, but got the {type(enable_dp_shard)}")
    if dp > world_size:
        raise ValueError(
            f"The value of dp must be smaller or equal word_size, but got the dp is {dp},"
            f"the world_size is {world_size}"
        )
    if sp > world_size:
        raise ValueError(
            f"The value of sp must be smaller or equal word_size, but got the dsp is {sp},"
            f"the world_size is {world_size}"
        )
    if enable_dp_shard:
        if dp * sp != world_size:
            raise ValueError(
                f"The product of dp and sp should be equal to total device number,"
                f"but got dp = {dp}, sp = {sp} and total device number = {world_size}"
            )

        seq_len = batch.shape[seq_dim]
        batch_sz = batch.shape[batch_dim]
        if seq_len % (2 * sp) != 0:
            raise ValueError(
                f"The sequence length of input batch is not divisible by 2*sp,"
                f"but got sequence length {seq_len} and sp is {sp}"
            )
        if batch_sz % dp != 0:
            raise ValueError(
                f"The batch size of input batch is not divisible by dp," f"but got batch_size {batch_sz} and dp is {dp}"
            )

        init_sp = get_sequence_parallel_world_size()
        if sp != init_sp:
            raise ValueError(
                f"The sp group is initialized as {init_sp}," f"but got different sp = {sp} in the input parameters"
            )

        sp_group_index = get_sequence_parallel_group_index()
        world_size = get_group_size()
        dp = world_size // sp
        if dp > 1:
            if batch_dim == 0:
                batch = batch.view(
                    dp,
                    batch.shape[batch_dim] // dp,
                    *batch.shape[(batch_dim + 1) :],
                )
            else:
                batch = batch.view(
                    *batch.shape[0:batch_dim],
                    dp,
                    batch.shape[batch_dim] // dp,
                    *batch.shape[(batch_dim + 1) :],
                )
            sp_group_index = Tensor([sp_group_index])
            batch = batch.index_select(batch_dim, sp_group_index).squeeze(batch_dim)

    if sp > 1:
        if seq_dim == 0:
            batch = batch.view(
                2 * sp,
                batch.shape[seq_dim] // (2 * sp),
                *batch.shape[(seq_dim + 1) :],
            )
        else:
            batch = batch.view(
                *batch.shape[0:seq_dim],
                2 * sp,
                batch.shape[seq_dim] // (2 * sp),
                *batch.shape[(seq_dim + 1) :],
            )

        index = Tensor([sp_rank, (2 * sp - sp_rank - 1)])
        batch = batch.index_select(seq_dim, index)

        if seq_dim == 0:
            batch = batch.view(-1, *batch.shape[(seq_dim + 2) :])
        else:
            batch = batch.view(*batch.shape[0:seq_dim], -1, *batch.shape[(seq_dim + 2) :])

    return batch
