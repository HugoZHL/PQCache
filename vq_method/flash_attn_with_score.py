import torch
import triton
import triton.language as tl
import math
import numpy as np

DTYPE = torch.float16

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i,
                    q,
                    K_block_ptr, V_block_ptr,
                    k_stride_n, v_stride_n,
                    start_m, qk_scale, N_CTX,
                    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
                    STAGE: tl.constexpr,
                    offs_m: tl.constexpr, offs_n: tl.constexpr,
                    ):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)

    K_block_ptr = K_block_ptr + lo * k_stride_n
    V_block_ptr = V_block_ptr + lo * v_stride_n

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float16)
        if STAGE == 2:
            k = tl.load(K_block_ptr, mask=(start_n + offs_n[None,:]) < N_CTX, other=0.0)
            qk += tl.dot(q, k)

            mask = (offs_m[:,None] < N_CTX) & (offs_m[:, None] >= (start_n + offs_n[None, :]))
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            k = tl.load(K_block_ptr)
            qk += tl.dot(q, k)

            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(V_block_ptr, mask=((start_n + offs_n[:,None]) < N_CTX), other=0.0)
        acc += tl.dot(p.to(tl.float16), v) 
        m_i = m_ij

        K_block_ptr = K_block_ptr + BLOCK_N * k_stride_n
        V_block_ptr = V_block_ptr + BLOCK_N * v_stride_n
    return acc, l_i, m_i

@triton.jit
def _attn_fwd_sum_score(l_i, m_i,
                        q,
                        K_block_ptr, F_ptrs,
                        k_stride_n, f_stride_n,
                        start_m, qk_scale, N_CTX,
                        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
                        STAGE: tl.constexpr, 
                        offs_m: tl.constexpr, offs_n: tl.constexpr,
                        ):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)

    K_block_ptr = K_block_ptr + lo * k_stride_n
    F_ptrs = F_ptrs + lo * f_stride_n

    
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if STAGE == 2:
            k = tl.load(K_block_ptr, mask=(start_n + offs_n[None,:]) < N_CTX, other=0.0)
            qk += tl.dot(q, k)

            mask = (offs_m[:,None] < N_CTX) & (offs_m[:, None] >= (start_n + offs_n[None, :]))
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            qk -= m_i[:, None]
        else:
            k = tl.load(K_block_ptr)
            qk += tl.dot(q, k)
            qk = qk * qk_scale - m_i[:, None] 

        eqk = tl.math.exp2(qk)
        eqk /= l_i[:,None]
        f_sum = tl.sum(eqk, 0)
        if STAGE == 2:
            tl.store(F_ptrs, f_sum, mask=(start_n + offs_n) < N_CTX)
        else:
            tl.store(F_ptrs, f_sum)

        K_block_ptr = K_block_ptr + BLOCK_N * k_stride_n
        F_ptrs = F_ptrs + BLOCK_N * f_stride_n

@triton.jit
def _attn_fwd_non_recent_max_score(l_i, m_i,
                                    q, 
                                    K_block_ptr, F_ptrs,
                                    k_stride_n, f_stride_n, 
                                    start_m, qk_scale, N_CTX,
                                    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,  
                                    STAGE: tl.constexpr, 
                                    offs_m: tl.constexpr, offs_n: tl.constexpr,  
                                    RECENT_CNT: tl.constexpr,
                                    ):
    
    if STAGE == 1: 
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2: 
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M) 

    K_block_ptr = K_block_ptr + lo * k_stride_n
    F_ptrs = F_ptrs + lo * f_stride_n

    
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if STAGE == 2:
            k = tl.load(K_block_ptr, mask=(start_n + offs_n[None,:]) < N_CTX, other=0.0)
            qk += tl.dot(q, k)

            mask = (offs_m[:,None] < N_CTX) & (offs_m[:, None] >= (start_n + offs_n[None, :]))
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            qk -= m_i[:, None]
        else:
            k = tl.load(K_block_ptr)
            qk += tl.dot(q, k)
            qk = qk * qk_scale - m_i[:, None] 

        eqk = tl.math.exp2(qk)
        eqk /= l_i[:,None]
        non_recent_mask = (start_n + offs_n)[None,:] <= (offs_m[:,None] - RECENT_CNT)
        eqk = tl.where(non_recent_mask, eqk, 0)
        f_max = tl.max(eqk, 0)
        if STAGE == 2:
            tl.store(F_ptrs, f_max, mask=(start_n + offs_n) < N_CTX)
        else:
            tl.store(F_ptrs, f_max)

        K_block_ptr = K_block_ptr + BLOCK_N * k_stride_n
        F_ptrs = F_ptrs + BLOCK_N * f_stride_n


@triton.jit
def _attn_fwd_prefill(Q, K, V, sm_scale, Out, F,
              stride_qz, stride_qh, stride_qm, stride_qk,  
              stride_kz, stride_kh, stride_kn, stride_kk,  
              stride_vz, stride_vh, stride_vk, stride_vn,  
              stride_oz, stride_oh, stride_om, stride_on,  
              stride_fz, stride_fh, stride_fm, stride_fn,  
              Z, H, N_CTX, 
              BLOCK_M: tl.constexpr,  
              BLOCK_DMODEL: tl.constexpr, 
              BLOCK_N: tl.constexpr, 
              SCORE_FUNC: tl.constexpr, 
              ):
    start_m = tl.program_id(0) 
    off_hz = tl.program_id(1) 
    off_z = off_hz // H 
    off_h = off_hz % H 
    
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    f_offset = off_z.to(tl.int64) * stride_fz + off_h.to(tl.int64) * stride_fh + start_m.to(tl.int64) * stride_fm

    D_idx = tl.arange(0,BLOCK_DMODEL)
    M_idx = start_m * BLOCK_M + tl.arange(0, BLOCK_M) 
    M_mask = (start_m * BLOCK_M + tl.arange(0,BLOCK_M)) < N_CTX 
    
    Q_PTR_OFFSET = q_offset + M_idx[:,None] * stride_qm + D_idx[None,:] * stride_qk
    Q_ptrs = Q + Q_PTR_OFFSET

    
    N_idx = tl.arange(0,BLOCK_N)
    K_PTR_OFFSET = k_offset + N_idx[None,:] * stride_kn + D_idx[:,None] * stride_kk 
    K_ptrs = K + K_PTR_OFFSET

    V_PTR_OFFSET = v_offset + N_idx[:,None] * stride_vk + D_idx[None,:] * stride_vn 
    V_ptrs = V + V_PTR_OFFSET

    O_PTR_OFFSET = o_offset + M_idx[:,None] * stride_om + D_idx[None,:] * stride_on
    O_ptrs = Out + O_PTR_OFFSET

    F_PTR_OFFSET = f_offset + N_idx * stride_fn
    F_ptrs = F + F_PTR_OFFSET

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf") 
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0 
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32) 
    
    qk_scale = sm_scale 
    qk_scale *= 1.44269504  
    
    q = tl.load(Q_ptrs, mask = M_mask[:, None], other=0.0)

    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_ptrs, V_ptrs, 
                                    stride_kn, stride_vk,
                                    start_m, qk_scale, N_CTX,  
                                    BLOCK_M, BLOCK_DMODEL, BLOCK_N,  
                                    1, offs_m, offs_n,  
                                    )

    tl.debug_barrier()
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_ptrs, V_ptrs,
                                    stride_kn, stride_vk,
                                    start_m, qk_scale, N_CTX,  
                                    BLOCK_M, BLOCK_DMODEL, BLOCK_N,  
                                    2, offs_m, offs_n, 
                                    )
    tl.debug_barrier()
    
    if SCORE_FUNC == 1:
        _attn_fwd_sum_score(l_i, m_i, q, K_ptrs, F_ptrs, stride_kn, stride_fn, 
                            start_m, qk_scale, N_CTX,
                            BLOCK_M, BLOCK_DMODEL, BLOCK_N,  
                            1, offs_m, offs_n, 
                            )
        tl.debug_barrier()
        _attn_fwd_sum_score(l_i, m_i, q, K_ptrs, F_ptrs, stride_kn, stride_fn, 
                            start_m, qk_scale, N_CTX,
                            BLOCK_M, BLOCK_DMODEL, BLOCK_N,  
                            2, offs_m, offs_n, 
                            )
    elif SCORE_FUNC == 2:
        RECENT_CNT = 32 
        _attn_fwd_non_recent_max_score(l_i, m_i, q, K_ptrs, F_ptrs, stride_kn, stride_fn, 
                            start_m, qk_scale, N_CTX,
                            BLOCK_M, BLOCK_DMODEL, BLOCK_N,  
                            1, offs_m, offs_n, 
                            RECENT_CNT
                            )
        tl.debug_barrier()
        _attn_fwd_non_recent_max_score(l_i, m_i, q, K_ptrs, F_ptrs, stride_kn, stride_fn, 
                            start_m, qk_scale, N_CTX,
                            BLOCK_M, BLOCK_DMODEL, BLOCK_N,  
                            2, offs_m, offs_n, 
                            RECENT_CNT
                            )
    
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    tl.store(O_ptrs, acc.to(Out.type.element_ty),mask = M_mask[:, None])


def flash_attn_with_score(Q, K, V, temperature = 1.0, phase="prefill", gumbel_adjustment=False, score_func="sum"):
    assert len(Q.shape) == 4 and len(K.shape) == 4
    assert Q.device == K.device and K.device == V.device
    
    num_stages = 3 
    num_warps = 8
    BLOCK_M = 128
    BLOCK_N = 64
    
    F = torch.zeros([Q.shape[0], Q.shape[1], math.ceil(K.shape[2]/BLOCK_M), K.shape[2]],
                            device = Q.device, dtype=torch.float32) 
    O = torch.empty_like(Q, device=Q.device)

    
    if torch.cuda.get_device_capability()[0] == 9:
        num_warps = 8
        num_stages = 7 if Q.shape[3] >= 64 else 3
    grid = (triton.cdiv(Q.shape[2], BLOCK_M), Q.shape[0]*Q.shape[1],1)
    
    if score_func == "sum":
        with torch.cuda.device(Q.device):
            _attn_fwd_prefill[grid](
                Q, K, V, temperature / math.sqrt(Q.shape[-1]), O, F,
                Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),  
                K.stride(0), K.stride(1), K.stride(2), K.stride(3),  
                V.stride(0), V.stride(1), V.stride(2), V.stride(3),  
                O.stride(0), O.stride(1), O.stride(2), O.stride(3),  
                F.stride(0), F.stride(1), F.stride(2), F.stride(3),
                Q.shape[0], Q.shape[1], N_CTX=Q.shape[2],  
                BLOCK_M=BLOCK_M,  
                BLOCK_N=BLOCK_N,  
                BLOCK_DMODEL=Q.shape[-1],  
                num_warps=num_warps,  
                num_stages=num_stages,  
                SCORE_FUNC=1
            )
        F = F.sum(dim=2) 
        if gumbel_adjustment:
            gumbel_noise = np.random.gumbel(size = F.shape)
            F = F + torch.tensor(gumbel_noise, device=F.device)
    elif score_func == "max":
        with torch.cuda.device(Q.device):
            _attn_fwd_prefill[grid](
                Q, K, V, temperature / math.sqrt(Q.shape[-1]), O, F,
                Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),  
                K.stride(0), K.stride(1), K.stride(2), K.stride(3),  
                V.stride(0), V.stride(1), V.stride(2), V.stride(3),  
                O.stride(0), O.stride(1), O.stride(2), O.stride(3),  
                F.stride(0), F.stride(1), F.stride(2), F.stride(3),
                Q.shape[0], Q.shape[1], N_CTX=Q.shape[2],  
                BLOCK_M=BLOCK_M,  
                BLOCK_N=BLOCK_N,  
                BLOCK_DMODEL=Q.shape[-1],  
                num_warps=num_warps,  
                num_stages=num_stages,  
                SCORE_FUNC=2
            )
        F = F.max(dim=2).values
    else:
        raise Exception("Invalid score func")
    return O, F

def test_op(Z, H, N_CTX, D_HEAD, phase, dtype=torch.float16):
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5))
    k = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5))
    v = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5))

    sm_scale = 1/ math.sqrt(D_HEAD)
    
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda")) 
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).to(DTYPE)
    
    ref_out = torch.matmul(p, v)
    from flash_attn import flash_attn_func
    ref_out_2 = flash_attn_func(
        q = q.transpose(1,2),
        k = k.transpose(1,2),
        v = v.transpose(1,2),
        softmax_scale=sm_scale,
        causal = True
    ).transpose(1,2)

    print(torch.norm(ref_out - ref_out_2))

    
    result, attn_sum = flash_attn_with_score(q, k, v, sm_scale, phase)

    assert torch.allclose(ref_out, result, atol=1e-2, rtol=0)
    print("ok")

def bench():
    pass
