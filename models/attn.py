import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys

class FullAttention(nn.Module):
    """
    Multi-head Attention Mechanism for the vanilla transformer.

    Parameters
    ----------
    scale: float, optional
        Used when computing softmax(scale K Q^T), by default 1/sqrt(D), where D is the number of variables of each instance.
    attention_dropout : float, optional
        The dropout probability applied to various parts of the layer, by default 0.1.
    output_attention : bool, optional
        Indicates if the attention scores should be returned as an output. If yes, we return (O, A), 
        where O is the output of the layer and A the matrix with the attention scores. Otherwise, 
        we return (O, None). By default False.

    Attributes
    ----------
    scale : float
        Used when computing softmax(scale K Q^T), by default 1/sqrt(D), where D is the number of variables of each instance.
    dropout : nn.Dropout
        Dropout for the attention layer. It is applied to the attention scores.
    output_attention : bool
        Indicates if the attention scores should be returned as an output. If yes, we return (O, A), 
        where O is the output of the layer and A the matrix with the attention scores. Otherwise, 
        we return (O, None). By default False.

    Methods
    -------
    config(src, window_size)
        Method to configure the attention layer in case window_size is needed to initialize the algorithm. 
        In this attention method, this information is not needed, so it does a pass.
    forward(src, queries, keys, values)
        Forward pass of the attention layer

    Examples
    --------
    Create an instance of the attention mechanism:

    >>> attn = FullAttention(output_attention=True)

    References
    ----------
    Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., ... & Polosukhin, I. (2017).
    Attention is all you need. In Advances in neural information processing systems (pp. 30-31).
    """
    def __init__(self, scale=None, attention_dropout=0.1, output_attention=False, **kwargs):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention

    def config(self, window_size = None):
        pass

    def forward(self, queries, keys, values):
        # B is the batch size, L is the number of queries (for instance, window size in the first layer), H is the number of heads, and E the dimension of each query.
        B, L, H, E = queries.shape
        # S is the number of values (for instance, window size in the first layer), D is the dimension of each value.
        _, S, _, D = values.shape
        # We scale the product Q K^T by multiplying by scale (usually 1/Sqrt(E))
        scale = self.scale or 1./math.sqrt(E)
        # Computes Q K^T, where Q = queries, K = keys (for all batches and heads). The result is parametrised in dimensions BxHxLxS, 
        # so we have a matrix with scalar products of size LxS for each instance of the batch and each head.
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # Scale the scores and apply softmax to compute the attention matrix
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        # Compute V as the sum A_ij values_j over j
        V = torch.einsum("bhls,bshd->blhd", A, values)
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    """
    Multi-head Attention Mechanism for the vanilla transformer.

    Parameters
    ----------
    factor : float, optional
        Factor used when computing the number of rows of Q used in ProbAttention, set to factor*log n. By default, set to 5.
    scale: float, optional
        Used when computing softmax(scale K Q^T), by default 1/sqrt(D), where D is the number of variables of each instance.
    attention_dropout : float, optional
        The dropout probability applied to various parts of the layer, by default 0.1.
    output_attention : bool, optional
        Indicates if the attention scores should be returned as an output. If yes, we return (O, A), 
        where O is the output of the layer and A the matrix with the attention scores. Otherwise, 
        we return (O, None). By default False.

    Attributes
    ----------
    factor : float
        Factor used when computing the number of rows of Q used in ProbAttention, set to factor*log n.
    scale : float
        Used when computing softmax(scale K Q^T), by default 1/sqrt(D), where D is the number of variables of each instance.
    dropout : nn.Dropout
        Dropout for the attention layer. It is applied to the attention scores.
    output_attention : bool
        Indicates if the attention scores should be returned as an output. If yes, we return (O, A), 
        where O is the output of the layer and A the matrix with the attention scores. Otherwise, 
        we return (O, None). By default False.

    Methods
    -------
    config(src, window_size)
        Method to configure the attention layer in case window_size is needed to initialize the algorithm. 
        In this attention method, this information is not needed, so it does a pass.
    
    _prob_QK(self, Q, K, sample_k, n_top)
        This is part of the algorithm for ProbAttention, see code for explanation.
    
    _get_initial_context(self, V, L_Q)
        This is part of the algorithm for ProbAttention, see code for explanation.
    
    _update_context(self, context_in, V, scores, index, L_Q)
        This is part of the algorithm for ProbAttention, see code for explanation.

    forward(src, queries, keys, values)
        Forward pass of the attention layer.

    Examples
    --------
    Create an instance of the attention mechanism:

    >>> attn = ProbAttention(output_attention=True)

    References
    ----------
    Zhou, Haoyi, et al. "Informer: Beyond efficient transformer for long sequence time-series forecasting." 
    Proceedings of the AAAI conference on artificial intelligence. Vol. 35. No. 12. 2021. 
    """
    def __init__(self, factor=5, scale=None, attention_dropout=0.1, output_attention=False, **kwargs):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def config(self, window_size = None, **kwargs):
        pass

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        """
        Q : queries of the attention mechanism.
        K : keys of the attention mechanism.
        sample_k : number of entries of K selected to compute the rows of Q that maximize the sparsity measurement M, see below.
        n_top : number of entries of the query matrix Q to be selected to compute attentions.

        Algorithm:
        1. Sample for each j in {1,2,...,L_Q}, sample_k rows from K, obtaining K_sample. 
           So K_sample[b,h,j,:,:] contains sample_k random rows of K.
        2. Calculate the scalar products the jth row of Q with the random rows of K in K_sample[b,h,j,:,:].
           There are O(n log n) scalar products in total, giving Q_K_sample, which has dimensions B x H x L_Q x sample_K
        3. Find the n_top rows of Q_K_sample that maximize the sparsity measurement M, given by
           M(q_i, K) = max_j{<q_i, k_j> } - sum_j <q_i, k_j> / L_K.
           This measurement follows from applying bounding the KL divergence between a row/vector of attentions and the uniform distribution for attentions.
           Here we approximate it, using only sample_k rows of K for each i (the ones used for Q_K).
           The list with n_top indexes of rows of Q obtained is denoted M_top.
        4. Multiply Q[M_top, :] by the transpose of K, to obtain the attention weights of the rows M_top of Q.

        Output: the scalar products Q_K, that will be used to compute the attentions of the attention mechanism, and the indexes of the rows of Q used, M_top.
        """
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # Calculate random indexes (matrix L_Qxsample_k) of elements of K
        # K_expand adds a new dimension to K, and copies this matrix, K, L_Q times in that dimension. This step does not take extra memory due to pytorch implementation.
        # index_sample is a matrix of dimensions L_Q x sample_k with random entries between 0 and L_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) 
       
        # Step 1 of algorithm (see above)
        # K_sample is a B x H x L_Q x sample_k x D tensor.
        # Each entry K_sample[b,h,l,:,:] is a list of size sample_k, and each element is a random row of K.
        # In total, there are L_Q such lists. This takes memory O(B H D n log n).
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :] 
        
        # Step 2 of algorithm (see above)
        # Q_K_sample is a tensor, such that Q_K_sample[b,h,j,:] has the scalar products of Q[b,h,j,:] and the rows of K in K_sample[b,h,j,:,:].
        # Q_K_sample has dimensions B x H x L_Q x sample_K
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        
        # Step 3 of algorithm (see above)
        # Find the Top_k query with sparisty measurement given by
        # M(q_i, K) = max_j{<q_i, k_j> } - sum_j <q_i, k_j> / L_K
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # Step 4 of algorithm (see above)
        # Use the reduced Q to calculate Q_K
        # Q_reduced has dimensions B x H x n_top x D
        # Q_K is the product of Q_reduced and K^T, has dimensions B x H x n_top x L_K
        Q_reduced = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]
        Q_K = torch.matmul(Q_reduced, K.transpose(-2, -1))
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        """
          Initialises the output of the attention mechanism to a tensor context of dimensions
          B x H x L_V x D, where context[b,h,l,:] is the average of all vectors in V[b,h,:,:].
        """
        B, H, L_V, D = V.shape

        # V_sum = V.sum(dim=-2)
        V_sum = V.mean(dim=-2)
        context = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()

        return context

    def _update_context(self, context_in, V, scores, index, L_Q):
        """
        context_in : a preliminary computation of the output of the attention mechanism. We update it with scores.
        scores : some of the scores of the full attention mechanism. More precisely, the input should be n_top rows of scores,
                 each row corresponding to a row of Q.
        index : indexes of the rows of Q used to compute scores. context_in will be updates on these indexes.
        L_Q : length of the matrix of queries Q.

        Computes the attentions for scores, and updates context_in with the corresponding outputs of the attention mechanism.
        """
        # B is the batch size, L_V is the number of values (for instance, window size in the first layer), H is the number of heads, and E the dimension of each query.
        B, H, L_V, D = V.shape

        attn = torch.softmax(scores, dim=-1) 
        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in) 
        
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values):
        """
        Applies a forward step of the ProbAttention mechanism.
        """
        # B is the batch size, L is the number of queries (for instance, window size in the first layer), H is the number of heads, and E the dimension of each query.
        B, L_Q, H, D = queries.shape
        # S is the number of values (for instance, window size in the first layer), D is the dimension of each value.
        _, L_K, _, _ = keys.shape

        # Move the heads to the first dimension.
        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        # Compute the number of rows of K that will be randomly selected in the method _prob_QK
        # U_part = min{factor*ln(L_k), L_k} (factor is called c in the paper)
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() 
        U_part = U_part if U_part<L_K else L_K
        # Computes the number of columns of Q that will be used when computing attentions.
        # u = min{factor*ln(L_q), L_q} (factor is called c in the paper)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() 
        u = u if u<L_Q else L_Q
        
        # Compute u rows of scores for the attention mechanism, using the heuristic explained in _prob_QK
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # Multiply by the scale factor
        scale = self.scale or 1./math.sqrt(D)
        scores_top = scores_top * scale
        # Get the initial context (tensor BxHxL_VxD where the entries [b,h,:,:] are the averages of those of V).
        context = self._get_initial_context(values, L_Q)
        # Update the context with the selected top_k queries and their scores
        context, attn = self._update_context(context, values, scores_top, index, L_Q)
        
        return context.transpose(2,1).contiguous(), attn

class LocalAttention(nn.Module):
    """
    Multi-head Attention Mechanism with local attention scores. 
    The idea is using only the keys ocurring just before a query, to exploit the time-structure of a time series.
    This accelerates the attention mechanism, running in time O(B H n l), where B is the number of batches, H is the
    number of heads, n is the length of the list of keys/queries and l is the number of keys we choose when computing 
    attention scores for each query. Formally, the output is a matrix (O_j), where
    O_j =  \sum_{i = j-l}ˆj \hat{a}_{ji} v_i
    and
    \hat{a}_{ji} = a_{ji} (the attention score of full attention mechanism) if j-l <= i <= j, and \hat{a}_{ji} = 0 otherwise.

    Normally we take l = \sqrt{n}, so the attention mechanism uses O(n \sqrt{n}) attention scores, instead of O(n^2). 
    The question is how we compute these scores efficiently, instead of first computing  a_{ji} and then using a mask. 
    This is resolved in our paper, and explained in the code below. The resulting complexity is O(n^2 / splits), see
    the description of the parameter splits and the paper or the code below.

    Parameters
    ----------
    neigh_size : Size of the local neighbourhood used in the local attention mechanism (called l in the description aboved).
    scale: float, optional
        Used when computing softmax(scale K Q^T), by default 1/sqrt(D), where D is the number of variables of each instance.
    attention_dropout : float, optional
        The dropout probability applied to various parts of the layer, by default 0.1.
    output_attention : bool, optional
        Indicates if the attention scores should be returned as an output. If yes, we return (O, A), 
        where O is the output of the layer and A the matrix with the attention scores. Otherwise, 
        we return (O, None). By default False.
    splits : int, optional
        Number of splits done in the local attention algorithm. This number determines in how many blocks the matrices Q and K are splitted
        when computing the local attention. he complexity of this attention mechanism is O(n^2 / splits). 
        A higher split number speeds up the algorithm, but if is splits is too high we may not be exploiting the paralelisation of pytorch. 
        A way to compensate for this, is increasing the batch size, see our paper for an execution time analysis.

    Attributes
    ----------
    scale : float
        Used when computing softmax(scale K Q^T), by default 1/sqrt(D), where D is the number of variables of each instance.
    attention_dropout : float
        The dropout probability applied to various parts of the layer, by default 0.1.
    output_attention : bool
        Indicates if the attention scores should be returned as an output. If yes, we return (O, A), 
        where O is the output of the layer and A the matrix with the attention scores. Otherwise, 
        we return (O, None). By default False.
    neigh_size : int
        Size of the local neighbourhood used in the local attention mechanism (called l in the description aboved).
    splits : int
        Number of splits done in the local attention algorithm. 
    local_mask_start : tensor of bools
        Mask for the first block of the local attention, see algorithm for details.
    local_mask_middle : tensor of bools
        Mask for the middle blocks of the local attention, see algorithm for details.
    local_mask_end : tensor of bools
        Mask for the last block of the local attention, see algorithm for details.
    window_size : int
        Size of the window of this attention layer, i.e., the first dimension of the queries/keys/values matrices.
    split_indexes : tuple of int
        Tuple with the indexes that determine the blocks of the local attention.
    
    Methods
    -------
    config(src, window_size)


    forward(src, queries, keys, values, attn_mask)
        Forward pass of the attention layer

    Examples
    --------
    Create an instance of the attention mechanism:

    >>> attn = LocalAttention(15, output_attention=True, splits = 10)

    References
    ----------
    Nacho Aguilera, Andres Herrera-Poyatos (2023).
    LocalTran: a transformer for time-series based on local attention mechanisms 
    """
    def __init__(self, neigh_size = None, scale=None, attention_dropout=0.1, output_attention=False, splits = None, device="cuda:0", **kwargs):
        super(LocalAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention
        self.neigh_size = neigh_size
        self.splits = splits 
        self.local_mask_start = None
        self.local_mask_middle = None
        self.local_mask_end = None
        self.window_size = None
        self.split_indexes = None
        self.device = device
 
    def config(self, window_size):
        """
        Computes the local_masks for local attention. We need 3 local masks, one for the start, one for the end, and one for middle blocks.
        The idea is masking the attention scores (before soft max is applied) by 
        M = 
        [1 0 0 0 0 ... 0]
        [1 1 0 0 0 ... 0]
        ...
        [1 1 ... 1 0 0... 0]
        [0 1 ... 1 1 0 ... 0]
        ...
        [0 0 ... 0 1 1 ... 1]
        where this matrix is window_size x window_size and the number of consecutive ones in each row is neigh_size.

        However, we don't want to apply this mask to the product Q K^T, as this leads to O(n^2) space and running time.
        Note that the matrix above is triangular, and, in fact, most entries are zero (only O(n neigh_size) are non-zero).
        The idea is exploiting this to avoid computing attention scores that are going to be masked to 0. In order to do so,
        we split Q and K into 'splits' blocks, and apply part of the mask above to each block. Following this idea, in this
        method we split the mask above into 'splits' blocks as follows
        Block 1:
        [1 0 0 0 0 ... 0]
        [1 1 0 0 0 ... 0]
        ...
        [1 1 ... 1 0 0 ... 0]
        [0 1 ... 1 1 0 ... 0]
        ...
        [0 0 ... 0 1 1 ... 1 ]
        where there are l1 = window_size / splits rows. Each row has l1 entries (as the rest of entries in the row are always 0).
        We need that neigh_size <=  window_size / splits, so each block contains all the ones of the 
        corresponding rows of the matrix M. 

        Blocks 2 to splits-1: 
        [1 1 ... 1 0  0... 0]
        [0 1 1 ... 1 0 ... 0]
        ...
        ...
        [0 0 ... 0 1 1 ... 1]
        There are l1 rows, each with l1+neigh_size entries.

        Block splits:
        [1 1 ... 1 0  0... 0]
        [0 1 1 ... 1 0 ... 0]
        ...
        ...
        [0 0 ... 0 1 1 ... 1]
        There are window_size - (splits -1) l1 rows (as window_size may not be divisible by splits).
        Each row has window_size - (splits -1) l1 + neigh_size entries.

        Note that the 3 type of blocks are different and the matrix M can be recovered by putting this blocks in order
        diagonally, so the diagonal of M is all ones, and filling in the gaps with zeroes.

        The code below pre-computes these masking blocks and makes sures neigh_size <=  window_size / splits. If the latter
        inequality does not hold because splits is too large, we set splits = window_size / neigh_size.

        There is one last caveat, and this is that since we apply the mask before softmax, instead of zeroes above we need -inf,
        so softmax only uses the entries of scores that have 1 in the mask.
        """
        self.window_size = window_size
        # Update neigh_size if needed
        if self.neigh_size == None:
            self.neigh_size = int(4*np.ceil(np.log(self.window_size))) #int(np.sqrt(self.window_size))
        # Update splits if needed
        if self.splits == None or self.splits < 0 or self.neigh_size > self.window_size / self.splits:
            self.splits = self.window_size // self.neigh_size
        l1 = window_size//self.splits # splits may not divide window_size, need to do this generalization and check if performance is affected
        #l1_rest = window_size - (self.splits-1)*l1
        l_end = (self.splits-1)*l1

        # Compute the full mask M described in the comment above.
        full_mask = torch.tril(torch.ones((window_size,window_size), dtype=torch.double)) - torch.tril(torch.ones((window_size,window_size), dtype=torch.double),diagonal=-self.neigh_size)
        full_mask = full_mask.to(self.device)
        #Replace 0 with -np.inf
        full_mask = full_mask.masked_fill(full_mask == 0, -sys.maxsize)
        full_mask = full_mask.masked_fill(full_mask == 1, 0)
        #np_mask = np.array([ [-np.inf]*j + [1]*self.neigh_size + [-np.inf]*(window_size-j) for j in torch.arange(0, window_size)])
        #np_mask = np_mask[0:window_size, self.neigh_size-1:window_size+self.neigh_size-1]
        #full_mask = torch.from_numpy(np_mask)

        # Local masks have dimensions:
        # l1 x l1
        # l1 x (l1+self.neigh_size)
        # (L-l_end) x (L-l_end+self.neigh_size)
        # We have a row for each input, and in each row we have the neigh_size ones corresponding to the attention scores computed.
        self.local_mask_start = full_mask[0:l1, 0:l1]
        self.local_mask_middle = full_mask[l1:2*l1, l1-self.neigh_size:2*l1]
        self.local_mask_end = self.local_mask_middle
        self.local_mask_end[window_size-l1*self.splits:,:] = -sys.maxsize

        # Indexes where we will split Q.
        #self.split_indexes = (j*l1 for j in torch.arange(1, self.splits))
        # Vectorize the line before
        #BUG: Should this be torch.arange(1, self.splits+1)*l1 ?
        split_indexes = []
        self.split_indexes_Q_end = None
        for j in torch.arange(0, self.splits):
            for i in torch.arange(0, l1):
                split_indexes.append(i+l1*j)
        self.split_indexes_Q = torch.tensor(split_indexes).reshape(self.splits, l1)

        if self.splits*l1 < window_size:
            split_indexes_end = []
            for i in torch.arange(0, l1):
                split_indexes_end.append(i+l1*self.splits)
            self.split_indexes_Q_end = torch.tensor(split_indexes_end)
            self.split_indexes_Q_end[window_size-self.splits*l1:] = torch.zeros((self.splits+1)*l1 - window_size)
        
        # Matrix SI such that for split i, the block i of K has the rows with index in SI[i,:]
        split_indexes = [] 
        self.split_indexes_KV_end = None
        for j in torch.arange(0, self.splits):
            for i in torch.arange(0, l1+self.neigh_size):
                split_indexes.append(i+l1*j-self.neigh_size)
        self.split_indexes_KV = torch.tensor(split_indexes).reshape(self.splits, l1+self.neigh_size)
        self.split_indexes_KV[0] = torch.cat((torch.zeros(self.neigh_size), torch.arange(0, l1)), 0)
        
        if self.splits*l1 < window_size:
            split_indexes_end = []
            for i in torch.arange(0, l1 + self.neigh_size):
                split_indexes_end.append(i+l1*self.splits-self.neigh_size)
            self.split_indexes_KV_end = torch.tensor(split_indexes_end)
            self.split_indexes_KV_end[window_size-self.splits*l1:] = torch.zeros((self.splits+1)*l1 + self.neigh_size - window_size)


    def forward(self, queries, keys, values):
        """
        Forward pass of the local attention mechanism.

        We compute Q K^T (.) M, where (.) is the Hadamard product and M is the full mask described in the method config.
        We do so efficiently by splitting Q and K in 'splits' blocks, and doing the operation above for each one of these blocks. 
        That is, we have the blocks of Q, for l1 = window_size // splits:
        Block 1:
        Row 0 of Q
        Row 1 of Q
        ...       
        Row l1-1 of Q

        Block 2:
        Row l1 of Q
        Row l1+1 of Q
        ...       
        Row 2*l1-1 of Q

        ...

        Block m ='splits':
        Row (m-1)l1 of Q
        Row (m-1)l1+1 of Q
        ...       
        Row window_size-1 of Q

        Note that here blocks are disjoint, i.e., each block has different rows of Q. 
        For K, that's not going to be the case. The blocks of K are, for m = splits:
        
        Block 1:
        Row 0 of K
        Row 1 of K
        ...       
        Row l1-1 of K

        Block 2:
        Row l1-neigh_size of K
        Row l1-neigh_size+1 of K
        ...       
        Row 2*l1-1-neigh_size of K

        ...

        Block m-1:
        Row (m-2)l1-neigh_size of K
        Row (m-2)l1-neigh_size+1 of K
        ...       
        Row (m-1)l1-1 of K

        Block m ='splits':
        Row (m-1)l1 - neigh_size of K
        Row (m-1)l1 - neigh_size +1 of K
        ...       
        Row window_size-1 of K

        The reason for repeated rows is that for the jth row of Q, we have to compute the attentions a_ji for i such that j-neigh_size <= i <= j,
        Since we perform the product of blocks of Q and blocks of K, if a block of Q has rows j, j+1, ..., j+k, then the corresponding block of K
        has to have the blocks j-neigh_size, ..., j+k.
        """
        # B is the batch size, L is the number of queries (for instance, window size in the first layer), H is the number of heads, and E the dimension of each query.
        B, L, H, E = queries.shape
        # S is the number of values (for instance, window size in the first layer), D is the dimension of each value.
        _, S, _, D = values.shape
        # We scale the product Q K^T by multiplying by scale (usually 1/Sqrt(E))
        self.scale = self.scale or 1./math.sqrt(E)

        # BUG: window size is 1 at some point of the execution, therefore this does not works. This could be due to
        # changing the model from autoencoder to predictor, but it should not happen. Check this.
        l1 = self.window_size//self.splits

        # Split queries
        Q_expand = queries.unsqueeze(0).expand(self.splits, B, L, H, E).permute(1,0,2,3,4)
        Q_split = Q_expand[:, torch.arange(self.splits).unsqueeze(1), self.split_indexes_Q, :].transpose(0,1)
        if self.splits*l1 < self.window_size:
            Q_end = queries[:,self.split_indexes_Q_end,:,:]
            Q_split = torch.cat((Q_split,Q_end.unsqueeze(0)), 0)
        
        # Split keys (here we need more rows per block to capture the attention scores of queries and those elements of keys at positions j l1 - neigh_size to j l1.
        K_expand = keys.unsqueeze(0).expand(self.splits, B, L, H, E).permute(1,0,2,3,4)
        K_split = K_expand[:, torch.arange(self.splits).unsqueeze(1), self.split_indexes_KV, :].transpose(0,1)
        K_split[0,:,:self.neigh_size,:,:] = 0
        if self.splits*l1 < self.window_size:
            K_end = keys[:,self.split_indexes_KV_end,:,:]
            K_split = torch.cat((K_split,K_end.unsqueeze(0)), 0)

        # Step 1 of algorithm (see above)
        # K_sample is a B x H x L_Q x sample_k x D tensor.
        # Each entry K_sample[b,h,l,:,:] is a list of size sample_k, and each element is a random row of K.
        # In total, there are L_Q such lists. This takes memory O(B H D n log n).

        V_expand = values.unsqueeze(0).expand(self.splits, B, L, H, E).permute(1,0,2,3,4)
        V_split = V_expand[:, torch.arange(self.splits).unsqueeze(1), self.split_indexes_KV, :].transpose(0,1)
        V_split[0,:,:self.neigh_size,:,:] = 0
        if self.splits*l1 < self.window_size:
            V_end = values[:,self.split_indexes_KV_end,:,:]
            V_split = torch.cat((V_split,V_end.unsqueeze(0)), 0)

        S = torch.einsum("sblhe,sbthe->sbhlt", Q_split, K_split)
        # BUG: local_mask_middle turns into empty at some point and this line fails.
        S_masked = S + self.local_mask_middle
        if self.splits*l1 < self.window_size:
            S_masked[-1, :, :, :] = S_masked[-1, :, :, :] + self.local_mask_end   
        S_masked = S_masked.nan_to_num(-sys.maxsize) # makes sure there is no np.inf * 0 product producing Nan, if so we send it back to -np.inf
        # Scale the scores and apply softmax to compute the attention matrix (not applied S1 locally at the moment...)
        A = self.dropout(torch.softmax(self.scale * S_masked, dim=-1))
        
        # Compute V as the sum A_ij values_j over j
        output = torch.einsum("sbhlt,sbthd->sblhd", A, V_split)
        output = torch.cat(torch.tensor_split(output, torch.arange(1,self.splits + 1*(self.splits*l1 < self.window_size)), 0),2).squeeze(0)
        output_refined = output[:,:self.window_size,:,:]
        if self.output_attention:
            #return (output.contiguous(), A)
            return (output_refined.contiguous(), torch.cat(A, 2)[:,:,self.neigh_size:self.window_size+self.neigh_size,:])
        else:
            return (output_refined.contiguous(), None)

class AttentionLayer(nn.Module):
    """
    Attention layer module: it configures an attention layer of one of the following types:
    - FullAttention
    - ProbAttention
    - LocalAttention
    This class contains the code that is common to any of the attention layers implemented in this library.

    Parameters
    ----------
    attention : nn.Module
        Instance of an attention mechanism.
    d_model : int
        Dimension of the input of the layer.
    n_heads: int
        Number of heads of the attention layer.
    d_keys : int, optional (by default None)
        First dimension of the keys matrix (i.e. number of keys).
    d_values : int, optional (by default None)
        First dimension of the value matrix (i.e. number of values).
    mix : bool, optional (False by default)
        If true, the output of the attention mechanism is represented as vectors of the form x_11, x_21, ..., x_12, ... 
        where x_ji is the ith variable of the jth head.
        Otherwise, the output of the attention mechanism is represented as vectors of the form x_11, x_12, ..., x_1d_values, x_21, ...

    Attributes
    ----------
    inner_attention : nn.Module
        Instance of an attention mechanism.
    query_projection : nn.Linear
        Full forward layer for the variables of the query matrix (operates only on the rows, 
        i.e. the input variables, not over the instances).
    key_projection : nn.Linear
        Full forward layer for the variables of the key matrix (operates only on the rows, 
        i.e. the input variables, not over the instances).
    value_projection : nn.Linear
        Full forward layer for the variables of the value matrix (operates only on the rows, 
        i.e. the input variables, not over the instances).        
    out_projection : nn.Linear
        Full forward layer for the the output matrix (linear combination of the output of the heads for each instance).        
    n_heads : int
        Number of heads in the attention layer.
    mix : bool
        If true, the output of the attention mechanism is represented as vectors of the form x_11, x_21, ..., x_12, ... 
        where x_ji is the ith variable of the jth head.
        Otherwise, the output of the attention mechanism is represented as vectors of the form x_11, x_12, ..., x_1d_values, x_21, ...  

    Methods
    -------
    forward(src, queries, keys, values, attn_mask)
        Forward pass of the attention layer.

    Examples
    --------
    Create an instance of the attentio layer:

    >>> attn = FullAttention(output_attention=True)
    >>> attn_layer = AttentionLayer(attn, 103, 5, d_keys = 20, d_values = 20)    
    """
    def __init__(self, attention, window_size, d_model, n_heads, d_keys=None, d_values=None, mix=False, batch_first=None, **kwargs):
        super(AttentionLayer, self).__init__()
        self.batch_first = batch_first

        #Dimension of keys and values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention

        # Linear layers for the queries, keys, values. They operate on each instance with d_model variables, 
        # and produce d_keys* n_heads new variables.
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        # Linear layer for the output of the attention mechanism, it merges the informaiton output by the n_heads, 
        # for each instance, producing d_model new variables.
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

        # Configurate the attention mechanism with the window size (only really needed for local attention) 
        self.inner_attention.config(window_size)

    def forward(self, queries, keys, values, **kwargs):
        # Extracts the dimensions of queries, keys, values.
        # B is the number of batches, L is the number of instances of the queries,
        # S is the number of instnaces of the keys.
        #  H is the number of heads in this layer.
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Applies the projections to queries, keys and values, rearranging the results as
        # tensors B x L x H x -1 (where -1 is d_keys or d_values).
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # Applies the attention mechanism.
        out, attn = self.inner_attention(
            queries,
            keys,
            values
        )

        # Places for each variable, all the outputs of the heads for that variable continuously.
        if self.mix:
            out = out.transpose(2,1)
            
        out = out.view(B, L, -1).contiguous()
        return self.out_projection(out), attn