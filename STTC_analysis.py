import sys
from pathlib import Path
import csv
import numpy as np
import time
import math

from numba import cuda, f8, u8, u1

tpb = 32
tpb_sq = tpb * tpb



@cuda.jit
def binaryOR(arr, res):
    x, y = cuda.grid(2)

    if x < arr.shape[0] and y < arr.shape[1]:
        res[x, y] |= arr[x, y]



@cuda.jit
def binaryAND(arr, res):
    x, y = cuda.grid(2)

    if x < res.shape[0] and y < res.shape[1]:
        res[x, y] &= arr[y]



@cuda.jit
def copy_array(arr, res):
    x, y = cuda.grid(2)

    if x < arr.shape[0] and y < arr.shape[1]:
        res[x, y] = arr[x, y]



@cuda.jit
def copy_array_flat(arr, res):
    x = cuda.grid(1)

    if x < arr.shape[0]:
        res[x] = arr[x]



@cuda.jit
def circular_shift(arr, res):
    x, y = cuda.grid(2)

    z = (y + x) % arr.shape[0]
    if x < res.shape[0] and y < res.shape[1]:
        res[x, y] = arr[z]



@cuda.jit
def circular_shift_rng(arr, sarr, res):
    x, y = cuda.grid(2)

    z = (y + sarr[x]) % arr.shape[0]
    if x < res.shape[0] and y < res.shape[1]:
        res[x, y] = arr[z]



@cuda.jit
def circular_shift_flat(arr, res):
    x = cuda.grid(1)

    size = arr.shape[0]
    i = x // size
    j = x % size
    z = (j + i + size) % size
    if x < res.shape[0]:
        res[i * size + j] = arr[z]



@cuda.jit
def sum_reduce_1d(arr, res):
    s_arr = cuda.shared.array(shape=tpb_sq, dtype=u8)
    
    x = cuda.grid(1)
    tx = cuda.threadIdx.x

    sum_OoG = 0
    for i in range(x, arr.shape[0], tpb_sq*cuda.gridDim.x):
        sum_OoG += arr[i]

    s_arr[tx] = sum_OoG

    cuda.syncthreads()

    s = 1
    while (s < cuda.blockDim.x):
        if (tx % (2 * s)) == 0:
            s_arr[tx] += s_arr[tx + s]
        
        cuda.syncthreads()

        s *= 2

    if tx == 0:
        res[cuda.blockIdx.x] = s_arr[0]



@cuda.jit
def sum_reduce_2d(arr, res):
    s_arr = cuda.shared.array(shape=(tpb, tpb), dtype=u8)
    
    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    if x >= arr.shape[0]:
        return

    sum_OoG = 0
    for j in range(y, arr.shape[1], tpb*cuda.gridDim.y):
        sum_OoG += arr[x, j]

    s_arr[tx, ty] = sum_OoG

    cuda.syncthreads()

    s = 1
    while (s < cuda.blockDim.y):
        if (ty % (2 * s)) == 0:
            s_arr[tx, ty] += s_arr[tx, ty + s]
        
        cuda.syncthreads()

        s *= 2

    if ty == 0:
        res[x, cuda.blockIdx.y] = s_arr[tx, 0]



@cuda.jit
def sum_reduce_2d_f8(arr, res):
    s_arr = cuda.shared.array(shape=(tpb, tpb), dtype=f8)
    
    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    if x >= arr.shape[0]:
        return

    sum_OoG = 0
    for j in range(y, arr.shape[1], tpb*cuda.gridDim.y):
        sum_OoG += arr[x, j]

    s_arr[tx, ty] = sum_OoG

    cuda.syncthreads()

    s = 1
    while (s < cuda.blockDim.y):
        if (ty % (2 * s)) == 0:
            s_arr[tx, ty] += s_arr[tx, ty + s]
        
        cuda.syncthreads()

        s *= 2

    if ty == 0:
        res[x, cuda.blockIdx.y] = s_arr[tx, 0]



@cuda.jit
def tiling_percentage(arr, sz, res):
    x = cuda.grid(1)

    if x < arr.shape[0]:
        res[x] = arr[x] / f8(sz)



@cuda.jit
def correlation(A, B, C):
    sA = cuda.shared.array(shape=(tpb, tpb), dtype=u1)
    sB = cuda.shared.array(shape=(tpb, tpb), dtype=u1)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    tmp = u8(0)
    for i in range((A.shape[1] + tpb - 1) // tpb):
        Ay = ty + i * tpb
        if x < A.shape[0] and Ay < A.shape[1]:
            sA[tx, ty] = A[x, Ay]
        else:
            sA[tx, ty] = 0
        Bx = tx + i * tpb
        if y < B.shape[0] and Bx < B.shape[1]:
            sB[ty, tx] = B[y, Bx]
        else:
            sB[ty, tx] = 0
            
        cuda.syncthreads()

        for j in range(tpb):
            tmp += sA[tx, j] & sB[ty, j]

        cuda.syncthreads()

    if x < C.shape[0] and y < C.shape[1]:
        C[x, y] = tmp



@cuda.jit
def correlation_percentage_x(Nc, N, P):
    x, y = cuda.grid(2)

    if x < P.shape[0] and y < P.shape[1]:
        P[x, y] = Nc[x, y] / f8(N[x])



@cuda.jit
def correlation_percentage_y(Nc, N, P):
    x, y = cuda.grid(2)

    if x < P.shape[0] and y < P.shape[1]:
        P[x, y] = Nc[x, y] / f8(N[y])



@cuda.jit
def STTC_pairs(T_Ap, P_B_Ap, T_Bm, P_A_Bm, STTC_AB):
    x, y = cuda.grid(2)

    if x < STTC_AB.shape[0] and y < STTC_AB.shape[1]:
        part1 = (P_A_Bm[x, y] - T_Bm[y]) / (1. - P_A_Bm[x, y] * T_Bm[y])
        part2 = (P_B_Ap[y, x] - T_Ap[x]) / (1. - P_B_Ap[y, x] * T_Ap[x])
        STTC_AB[x, y] = .5 * (part1 + part2)



@cuda.jit
def STTC_triplets(T_redAp, P_redAp_B, T_Bm, P_Bm_redA, STTC_redAB):
    x, y = cuda.grid(2)

    if x < STTC_redAB.shape[0] and y < STTC_redAB.shape[1]:
        part1 = (P_Bm_redA[x, y] - T_Bm[x]) / (1. - P_Bm_redA[x, y] * T_Bm[x])
        part2 = (P_redAp_B[y, x] - T_redAp[y]) / (1. - P_redAp_B[y, x] * T_redAp[y])
        STTC_redAB[x, y] = .5 * (part1 + part2)



@cuda.jit
def binary_filter_nonzero_1d(arr, res):
    x = cuda.grid(1)

    if x < res.shape[0]:
        if arr[x] > 0:
            res[x] = 1
        else:
            res[x] = 0



@cuda.jit
def filter_nonzero_2d(arr, sz, res):
    x, y = cuda.grid(2)

    if x < res.shape[0] and y < res.shape[1]:
        if sz[y] > 0:
            res[x, y] = arr[x, y]
        else:
            res[x, y] = 0



@cuda.jit
def deviation(samples, s_mean, res):
    x, y = cuda.grid(2)

    if x < res.shape[0] and y < res.shape[1]:
        res[x, y] = pow(samples[x, y] - s_mean[x], 2.0)



@cuda.jit
def sqrt_mean(arr, sz, res):
    x = cuda.grid(1)

    if x < arr.shape[0]:
        res[x] = pow(arr[x] / f8(sz), 0.5)



@cuda.jit
def zscore(x_arr, m_arr, s_arr, res):
    x = cuda.grid(1)

    if x < x_arr.shape[0]:
        res[x] = (x_arr[x] - m_arr[x]) / f8(s_arr[x])



def STTC_pairs_analysis_gpu(IDs, A, Dt, filename):
    start_time = time.time()

    N = A.shape[0]
    F = A.shape[1]
    Shifts = F - 1
    full_filename = filename.replace('.csv', '_all-shifts_' + str(Dt) + '-dt_pairs.csv')

    TPB = (tpb, tpb)

    BPG_N = math.ceil(N / tpb)
    BPG_F = math.ceil(F / tpb)
    BPG_Sm = math.ceil(Shifts / tpb)
    BPG_NxF = (BPG_N, BPG_F)
    BPG_Nx1 = (BPG_N, 1)
    BPG_FxF = (BPG_F, BPG_F)
    BPG_Fx1 = (BPG_F, 1)
    BPG_FxN = (BPG_F, BPG_N)
    BPG_NxSm = (BPG_N, BPG_Sm)

    d_A = cuda.to_device(A)
    d_A_r = cuda.device_array((N, F), dtype=np.uint64)

    d_Ap = cuda.device_array((N, F), dtype=np.uint8)
    d_Ap_r = cuda.device_array((N, F), dtype=np.uint64)
    d_T_Ap = cuda.device_array(N, dtype=np.float64)

    d_B = cuda.device_array((F, F), dtype=np.uint8)
    d_B_r = cuda.device_array((F, F), dtype=np.uint64)

    d_Bm = cuda.device_array((F, F), dtype=np.uint8)
    d_Bm_r = cuda.device_array((F, F), dtype=np.uint64)
    d_T_Bm = cuda.device_array(F, dtype=np.float64)

    d_N_B_Ap = cuda.device_array((F, N), dtype=np.uint64)
    d_N_A_Bm = cuda.device_array((N, F), dtype=np.uint64)

    d_P_B_Ap = cuda.device_array((F, N), dtype=np.float64)
    d_P_A_Bm = cuda.device_array((N, F), dtype=np.float64)

    d_STTC_AB = cuda.device_array((N, F), dtype=np.float64)

    d_CG_r = cuda.device_array((N, Shifts), dtype=np.float64)
    d_CGd = cuda.device_array((N, Shifts), dtype=np.float64)
    d_CGd_r = cuda.device_array((N, Shifts), dtype=np.float64)

    d_CGmean = cuda.device_array(N, dtype=np.float64)
    d_CGstdev = cuda.device_array(N, dtype=np.float64)
    d_CGzscore = cuda.device_array(N, dtype=np.float64)

    # find space occupied by spikes of neuron A in timeseries
    sum_reduce_2d[BPG_NxF, TPB](d_A, d_A_r)
    #cuda.synchronize()
    sum_reduce_2d[BPG_Nx1, TPB](d_A_r, d_A_r)
    #cuda.synchronize()

    # create tiles for neuron A
    copy_array[BPG_NxF, TPB](d_A, d_Ap)
    #cuda.synchronize()
    for t in range(1, Dt+1):
        #d_Ap[:, t:] = d_Ap[:, t:] | d_A[:, :-t]
        BPG_Fm = math.ceil((F - t) / tpb)
        BPG_NxFm = (BPG_N, BPG_Fm)
        binaryOR[BPG_NxFm, TPB](d_A[:, :-t], d_Ap[:, t:])
        #cuda.synchronize()

    # find space occupied by tiles of neuron A in timeseries
    sum_reduce_2d[BPG_NxF, TPB](d_Ap, d_Ap_r)
    #cuda.synchronize()
    sum_reduce_2d[BPG_Nx1, TPB](d_Ap_r, d_Ap_r)
    #cuda.synchronize()
    tiling_percentage[math.ceil(N / tpb_sq), tpb_sq](d_Ap_r[:, 0], F, d_T_Ap)
    #cuda.synchronize()
    
    # some timing prints
    some = 100 if N > 99 else 10

    some_tmp = some
    some_percent = math.ceil(N / some)
    some_sum = some_percent

    curr_time = time.time()
    some_percent_time = curr_time - start_time
    prev_time = curr_time
    (mins, secs) = divmod(some_percent_time, 60.)
    print("%i/%i - %im%.3fs" % (some - some_tmp, some, mins, secs))

    with open(full_filename, 'w') as f:
        f.write('NeuronA,NeuronB,STTC,CtrlGrpMean,CtrlGrpStDev,NullSTTC,Zscore\n')
    
    # for each neuron in dataset
    for n in range(N):
        #cuda.synchronize()

        # some timing prints
        if n == some_sum:
            some_tmp -= 1
            some_percent = math.ceil((N - some_sum) / some_tmp)
            some_sum += some_percent

            curr_time = time.time()
            some_percent_time = curr_time - prev_time
            prev_time = curr_time
            (mins, secs) = divmod(some_percent_time, 60.)
            print("%i/%i - %im%.3fs" % (some - some_tmp, some, mins, secs))

            pass
        
        # create the null array for neuron B
        circular_shift[BPG_FxF, TPB](d_A[n, :], d_B)
        #d_B = d_B.ravel(order='C')
        #circular_shift_flat[math.ceil(F * F / tpb_sq), tpb_sq](d_A[n, :], d_B)
        #d_B = d_B.reshape((F, F), order='C')
        #cuda.synchronize()
        
        # find space occupied by spikes of neuron B in timeseries
        sum_reduce_2d[BPG_FxF, TPB](d_B, d_B_r)
        #cuda.synchronize()
        sum_reduce_2d[BPG_Fx1, TPB](d_B_r, d_B_r)
        #cuda.synchronize()

        # find correlation between timeseries of neuron B and tiles of neuron A
        correlation[BPG_FxN, TPB](d_B, d_Ap, d_N_B_Ap)
        #cuda.synchronize()
        correlation_percentage_x[BPG_FxN, TPB](d_N_B_Ap, d_B_r[:, 0], d_P_B_Ap)
        #cuda.synchronize()
        
        # create tiles for neuron B
        copy_array[BPG_FxF, TPB](d_B, d_Bm)
        #cuda.synchronize()
        for t in range(1, Dt+1):
            #d_Bm[:, :-t] = d_Bm[:, :-t] | d_B[:, t:]
            BPG_Fm = math.ceil((F - t) / tpb)
            BPG_FxFm = (BPG_F, BPG_Fm)
            binaryOR[BPG_FxFm, TPB](d_B[:, t:], d_Bm[:, :-t])
            #cuda.synchronize()
        
        # find space occupied by tiles of neuron B in timeseries
        sum_reduce_2d[BPG_FxF, TPB](d_Bm, d_Bm_r)
        #cuda.synchronize()
        sum_reduce_2d[BPG_Fx1, TPB](d_Bm_r, d_Bm_r)
        #cuda.synchronize()
        tiling_percentage[math.ceil(F / tpb_sq), tpb_sq](d_Bm_r[:, 0], F, d_T_Bm)
        #cuda.synchronize()

        # find correlation between timeseries of neuron A and tiles of neuron B
        correlation[BPG_NxF, TPB](d_A, d_Bm, d_N_A_Bm)
        #cuda.synchronize()
        correlation_percentage_x[BPG_NxF, TPB](d_N_A_Bm, d_A_r[:, 0], d_P_A_Bm)
        #cuda.synchronize()

        # calculate STTC
        STTC_pairs[BPG_NxF, TPB](d_T_Ap, d_P_B_Ap, d_T_Bm, d_P_A_Bm, d_STTC_AB)
        #cuda.synchronize()
        
        h_STTC_AB = d_STTC_AB[:, 0].copy_to_host()
        h_STTC_Null = d_STTC_AB[:, 1].copy_to_host()

        # calculate CtrlGrpMean
        sum_reduce_2d_f8[BPG_NxSm, TPB](d_STTC_AB[:, 1:], d_CG_r)
        sum_reduce_2d_f8[BPG_Nx1, TPB](d_CG_r, d_CG_r)
        tiling_percentage[math.ceil(N / tpb_sq), tpb_sq](d_CG_r[:, 0], Shifts, d_CGmean)
        #cuda.synchronize()
        
        h_CGmean = d_CGmean.copy_to_host()

        # calculate CtrlGrpStDev
        deviation[BPG_NxSm, TPB](d_STTC_AB[:, 1:], d_CGmean, d_CGd)
        sum_reduce_2d_f8[BPG_NxSm, TPB](d_CGd, d_CGd_r)
        sum_reduce_2d_f8[BPG_Nx1, TPB](d_CGd_r, d_CGd_r)
        sqrt_mean[math.ceil(N / tpb_sq), tpb_sq](d_CGd_r[:, 0], Shifts, d_CGstdev)
        #cuda.synchronize()
        
        h_CGstdev = d_CGstdev.copy_to_host()

        zscore[math.ceil(N / tpb_sq), tpb_sq](d_STTC_AB[:, 0], d_CGmean, d_CGstdev, d_CGzscore)
        #cuda.synchronize()
        
        h_CGzscore = d_CGzscore.copy_to_host()

        h_nA = np.copy(IDs)

        h_nB = np.full(N, IDs[n])

        h_arr = np.column_stack((h_nA, h_nB, h_STTC_AB, h_CGmean, h_CGstdev, h_STTC_Null, h_CGzscore))

        h_arr = np.delete(h_arr, n, 0)

        with open(full_filename, 'a') as f:
            np.savetxt(f, h_arr, fmt='%d,%d,%.10f,%.10f,%.10f,%.10f,%.10f')

        continue

    # some timing prints
    curr_time = time.time()
    some_percent_time = curr_time - prev_time
    (mins, secs) = divmod(some_percent_time, 60.)
    print("%i/%i - %im%.3fs" % (some, some, mins, secs))

    pass



def STTC_pairs_analysis_gpu_rng(IDs, A, Dt, Shifts, filename):
    start_time = time.time()

    N = A.shape[0]
    F = A.shape[1]
    S = 1 + Shifts
    full_filename = filename.replace('.csv', '_' + str(Shifts) + '-shifts_' + str(Dt) + '-dt_pairs.csv')

    TPB = (tpb, tpb)

    BPG_N = math.ceil(N / tpb)
    BPG_F = math.ceil(F / tpb)
    BPG_S = math.ceil(S / tpb)
    BPG_Sm = math.ceil(Shifts / tpb)
    BPG_NxF = (BPG_N, BPG_F)
    BPG_Nx1 = (BPG_N, 1)
    BPG_SxF = (BPG_S, BPG_F)
    BPG_Sx1 = (BPG_S, 1)
    BPG_SxN = (BPG_S, BPG_N)
    BPG_NxS = (BPG_N, BPG_S)
    BPG_NxSm = (BPG_N, BPG_Sm)

    d_A = cuda.to_device(A)
    d_A_r = cuda.device_array((N, F), dtype=np.uint64)

    d_Ap = cuda.device_array((N, F), dtype=np.uint8)
    d_Ap_r = cuda.device_array((N, F), dtype=np.uint64)
    d_T_Ap = cuda.device_array(N, dtype=np.float64)

    d_B = cuda.device_array((S, F), dtype=np.uint8)
    d_B_r = cuda.device_array((S, F), dtype=np.uint64)

    d_Bm = cuda.device_array((S, F), dtype=np.uint8)
    d_Bm_r = cuda.device_array((S, F), dtype=np.uint64)
    d_T_Bm = cuda.device_array(S, dtype=np.float64)

    d_N_B_Ap = cuda.device_array((S, N), dtype=np.uint64)
    d_N_A_Bm = cuda.device_array((N, S), dtype=np.uint64)

    d_P_B_Ap = cuda.device_array((S, N), dtype=np.float64)
    d_P_A_Bm = cuda.device_array((N, S), dtype=np.float64)

    d_STTC_AB = cuda.device_array((N, S), dtype=np.float64)

    d_CG_r = cuda.device_array((N, Shifts), dtype=np.float64)
    d_CGd = cuda.device_array((N, Shifts), dtype=np.float64)
    d_CGd_r = cuda.device_array((N, Shifts), dtype=np.float64)

    d_CGmean = cuda.device_array(N, dtype=np.float64)
    d_CGstdev = cuda.device_array(N, dtype=np.float64)
    d_CGzscore = cuda.device_array(N, dtype=np.float64)

    # find space occupied by spikes of neuron A in timeseries
    sum_reduce_2d[BPG_NxF, TPB](d_A, d_A_r)
    #cuda.synchronize()
    sum_reduce_2d[BPG_Nx1, TPB](d_A_r, d_A_r)
    #cuda.synchronize()

    # create tiles for neuron A
    copy_array[BPG_NxF, TPB](d_A, d_Ap)
    #cuda.synchronize()
    for t in range(1, Dt+1):
        #d_Ap[:, t:] = d_Ap[:, t:] | d_A[:, :-t]
        BPG_Fm = math.ceil((F - t) / tpb)
        BPG_NxFm = (BPG_N, BPG_Fm)
        binaryOR[BPG_NxFm, TPB](d_A[:, :-t], d_Ap[:, t:])
        #cuda.synchronize()

    # find space occupied by tiles of neuron A in timeseries
    sum_reduce_2d[BPG_NxF, TPB](d_Ap, d_Ap_r)
    #cuda.synchronize()
    sum_reduce_2d[BPG_Nx1, TPB](d_Ap_r, d_Ap_r)
    #cuda.synchronize()
    tiling_percentage[math.ceil(N / tpb_sq), tpb_sq](d_Ap_r[:, 0], F, d_T_Ap)
    #cuda.synchronize()
    
    # some timing prints
    some = 100 if N > 99 else 10

    some_tmp = some
    some_percent = math.ceil(N / some)
    some_sum = some_percent

    curr_time = time.time()
    some_percent_time = curr_time - start_time
    prev_time = curr_time
    (mins, secs) = divmod(some_percent_time, 60.)
    print("%i/%i - %im%.3fs" % (some - some_tmp, some, mins, secs))

    with open(full_filename, 'w') as f:
        f.write('NeuronA,NeuronB,STTC,CtrlGrpMean,CtrlGrpStDev,NullSTTC,Zscore\n')
    
    # for each neuron in dataset
    for n in range(N):
        #cuda.synchronize()

        # some timing prints
        if n == some_sum:
            some_tmp -= 1
            some_percent = math.ceil((N - some_sum) / some_tmp)
            some_sum += some_percent

            curr_time = time.time()
            some_percent_time = curr_time - prev_time
            prev_time = curr_time
            (mins, secs) = divmod(some_percent_time, 60.)
            print("%i/%i - %im%.3fs" % (some - some_tmp, some, mins, secs))

            pass
        
        # create the null array for neuron B
        h_shift_ary = np.random.randint(1, high=F, size=S)
        h_shift_ary[0] = 0
        d_shift_ary = cuda.to_device(h_shift_ary)
        circular_shift_rng[BPG_SxF, TPB](d_A[n, :], d_shift_ary, d_B)
        #cuda.synchronize()
        
        # find space occupied by spikes of neuron B in timeseries
        sum_reduce_2d[BPG_SxF, TPB](d_B, d_B_r)
        #cuda.synchronize()
        sum_reduce_2d[BPG_Sx1, TPB](d_B_r, d_B_r)
        #cuda.synchronize()

        # find correlation between timeseries of neuron B and tiles of neuron A
        correlation[BPG_SxN, TPB](d_B, d_Ap, d_N_B_Ap)
        #cuda.synchronize()
        correlation_percentage_x[BPG_SxN, TPB](d_N_B_Ap, d_B_r[:, 0], d_P_B_Ap)
        #cuda.synchronize()
        
        # create tiles for neuron B
        copy_array[BPG_SxF, TPB](d_B, d_Bm)
        #cuda.synchronize()
        for t in range(1, Dt+1):
            #d_Bm[:, :-t] = d_Bm[:, :-t] | d_B[:, t:]
            BPG_Fm = math.ceil((F - t) / tpb)
            BPG_SxFm = (BPG_S, BPG_Fm)
            binaryOR[BPG_SxFm, TPB](d_B[:, t:], d_Bm[:, :-t])
            #cuda.synchronize()
        
        # find space occupied by tiles of neuron B in timeseries
        sum_reduce_2d[BPG_SxF, TPB](d_Bm, d_Bm_r)
        #cuda.synchronize()
        sum_reduce_2d[BPG_Sx1, TPB](d_Bm_r, d_Bm_r)
        #cuda.synchronize()
        tiling_percentage[math.ceil(S / tpb_sq), tpb_sq](d_Bm_r[:, 0], F, d_T_Bm)
        #cuda.synchronize()

        # find correlation between timeseries of neuron A and tiles of neuron B
        correlation[BPG_NxS, TPB](d_A, d_Bm, d_N_A_Bm)
        #cuda.synchronize()
        correlation_percentage_x[BPG_NxS, TPB](d_N_A_Bm, d_A_r[:, 0], d_P_A_Bm)
        #cuda.synchronize()

        # calculate STTC
        STTC_pairs[BPG_NxS, TPB](d_T_Ap, d_P_B_Ap, d_T_Bm, d_P_A_Bm, d_STTC_AB)
        #cuda.synchronize()
        
        h_STTC_AB = d_STTC_AB[:, 0].copy_to_host()
        h_STTC_Null = d_STTC_AB[:, 1].copy_to_host()

        # calculate CtrlGrpMean
        sum_reduce_2d_f8[BPG_NxSm, TPB](d_STTC_AB[:, 1:], d_CG_r)
        sum_reduce_2d_f8[BPG_Nx1, TPB](d_CG_r, d_CG_r)
        tiling_percentage[math.ceil(N / tpb_sq), tpb_sq](d_CG_r[:, 0], Shifts, d_CGmean)
        #cuda.synchronize()
        
        h_CGmean = d_CGmean.copy_to_host()

        # calculate CtrlGrpStDev
        deviation[BPG_NxSm, TPB](d_STTC_AB[:, 1:], d_CGmean, d_CGd)
        sum_reduce_2d_f8[BPG_NxSm, TPB](d_CGd, d_CGd_r)
        sum_reduce_2d_f8[BPG_Nx1, TPB](d_CGd_r, d_CGd_r)
        sqrt_mean[math.ceil(N / tpb_sq), tpb_sq](d_CGd_r[:, 0], Shifts, d_CGstdev)
        #cuda.synchronize()
        
        h_CGstdev = d_CGstdev.copy_to_host()

        zscore[math.ceil(N / tpb_sq), tpb_sq](d_STTC_AB[:, 0], d_CGmean, d_CGstdev, d_CGzscore)
        #cuda.synchronize()
        
        h_CGzscore = d_CGzscore.copy_to_host()

        h_nA = np.copy(IDs)

        h_nB = np.full(N, IDs[n])

        h_arr = np.column_stack((h_nA, h_nB, h_STTC_AB, h_CGmean, h_CGstdev, h_STTC_Null, h_CGzscore))

        h_arr = np.delete(h_arr, n, 0)

        with open(full_filename, 'a') as f:
            np.savetxt(f, h_arr, fmt='%d,%d,%.10f,%.10f,%.10f,%.10f,%.10f')

        continue

    # some timing prints
    curr_time = time.time()
    some_percent_time = curr_time - prev_time
    (mins, secs) = divmod(some_percent_time, 60.)
    print("%i/%i - %im%.3fs" % (some, some, mins, secs))

    pass



class from_device:
    enabled = 0
    pass



def STTC_pairs_analysis_cpu_no_null(A, Dt, res):
    start_time = time.time()

    N = A.shape[0]
    F = A.shape[1]

    A_r = np.sum(A, dtype=np.uint64, axis=1)

    Ap = np.copy(A)
    for t in range(1, Dt+1):
        Ap[:, t:] |= A[:, :-t]
        continue

    Ap_r = np.sum(Ap, dtype=np.uint64, axis=1)

    T_Ap = Ap_r / float(F)

    B = A

    B_r = A_r

    Bm = np.copy(B)
    for t in range(1, Dt+1):
        Bm[:, :-t] |= B[:, t:]
        continue

    Bm_r = np.sum(Bm, dtype=np.uint64, axis=1)

    T_Bm = Bm_r / float(F)

    N_Ap_B = np.empty((N, N), dtype=np.uint64)

    P_Ap_B = np.empty((N, N), dtype=np.float64)

    N_A_Bm = np.empty((N, N), dtype=np.uint64)

    P_A_Bm = np.empty((N, N), dtype=np.float64)

    STTC_AB = np.empty((N, N), dtype=np.float64)

    for i in range(N):
        Ai = A[i, :]
        #print("Ai:\n{}\n".format(Ai))

        Ai_r = A_r[i]
        #print("Ai_r:\n{}\n".format(Ai_r))
        
        Aip = Ap[i, :]
        #print("Aip:\n{}\n".format(Aip))

        Aip_r = Ap_r[i]
        #print("Aip_r:\n{}\n".format(Aip_r))

        T_Aip = T_Ap[i]
        #print("T_Aip:\n{}\n".format(T_Aip))

        for j in range(N):
            Bj = B[j, :]
            #print("Bj:\n{}\n".format(Bj))

            Bj_r = B_r[j]
            #print("Bj_r:\n{}\n".format(Bj_r))

            B_Ap = np.empty_like(Bj)
            B_Ap[:] = Bj[:] & Aip[:]
            #print("B_Ap:\n{}\n".format(B_Ap))
            N_Ap_B[i, j] = np.sum(B_Ap, dtype=np.uint64)
            #print("N_B_Ap:\n{}\n".format(N_B_Ap[i, j]))
            if Bj_r == 0:
                #print("res.h_P_Ap_B[i, j]: {}".format(res.h_P_Ap_B[i, j]))
                Bli = i
                Blj = j
                P_Ap_B[i, j] = 0.
            else:
                P_Ap_B[i, j] = N_Ap_B[i, j] / float(Bj_r)
            #print("P_B_Ap:\n{}\n".format(P_B_Ap[i, j]))
            
            Bjm = Bm[j, :]
            #print("Bjm:\n{}\n".format(Bjm))

            Bjm_r = Bm_r[j]
            #print("Bjm_r:\n{}\n".format(Bjm_r))

            T_Bjm = T_Bm[j]
            #print("T_Bjm:\n{}\n".format(T_Bjm))

            A_Bm = np.empty_like(Ai)
            A_Bm[:] = Ai[:] & Bjm[:]
            #print("A_Bm:\n{}\n".format(A_Bm))
            N_A_Bm[i, j] = np.sum(A_Bm, dtype=np.uint64)
            #print("N_A_Bm:\n{}\n".format(N_A_Bm[i, j]))
            if Ai_r == 0:
                #print("res.h_P_A_Bm[i, j]: {}".format(res.h_P_A_Bm[i, j]))
                Ali = i
                Alj = j
                P_A_Bm[i, j] = 0.
            else:
                P_A_Bm[i, j] = N_A_Bm[i, j] / float(Ai_r)
            #print("P_A_Bm:\n{}\n".format(P_A_Bm[i, j]))

            part1 = (P_A_Bm[i, j] - T_Bjm) / (1. - P_A_Bm[i, j] * T_Bjm)
            part2 = (P_Ap_B[i, j] - T_Aip) / (1. - P_Ap_B[i, j] * T_Aip)
            STTC_AB[i, j] = .5 * (part1 + part2)

            continue

        continue

    analysis_time = time.time() - start_time
    (mins, secs) = divmod(analysis_time, 60.)
    print("CPU Computational Analysis: %im%.3fs\n(1 thread)" % (mins, secs))

    if res.enabled == 1:
        print("A_r same? {}".format(np.array_equal(A_r, res.h_A_r[:, 0])))
        #print("A_r:\n{}\n".format(A_r))
        #print("res.h_A_r:\n{}\n".format(res.h_A_r[:, 0]))

        print("Ap same? {}".format(np.array_equal(Ap, res.h_Ap)))

        print("Ap_r same? {}".format(np.array_equal(Ap_r, res.h_Ap_r[:, 0])))

        print("T_Ap same? {}".format(np.array_equal(T_Ap, res.h_T_Ap)))

        print("Bm same? {}".format(np.array_equal(Bm, res.h_Bm)))
        #print("B:\n{}\n".format(B))
        #print("Bm:\n{}\n".format(Bm))
        #print("res.h_Bm:\n{}\n".format(res.h_Bm))
        #print("B:\n{}\n".format(B[0, :50].tolist()))
        #print("Bm:\n{}\n".format(Bm[0, :50].tolist()))
        #print("res.h_Bm:\n{}\n".format(res.h_Bm[0, :50].tolist()))

        print("Bm_r same? {}".format(np.array_equal(Bm_r, res.h_Bm_r[:, 0])))
        #print("Bm_r:\n{}\n".format(Bm_r))
        #print("res.h_Bm_r:\n{}\n".format(res.h_Bm_r[:, 0]))

        print("T_Bm same? {}".format(np.array_equal(T_Bm, res.h_T_Bm)))
        
        print("N_Ap_B same? {}".format(np.array_equal(N_Ap_B, res.h_N_Ap_B)))
        
        #print("res.h_P_A_Bm[i, j]: {}".format(res.h_P_A_Bm[Ali, Alj]))
        #print("P_A_Bm[i, j]:   {}".format(P_A_Bm[Ali, Alj]))
        
        print("P_Ap_B same? {}".format(np.array_equal(P_Ap_B, res.h_P_Ap_B)))
        
        print("N_A_Bm same? {}".format(np.array_equal(N_A_Bm, res.h_N_A_Bm)))
        
        print("P_A_Bm same? {}".format(np.array_equal(P_A_Bm, res.h_P_A_Bm)))
        
        #print("res.h_STTC_AB[i, j]: {}".format(res.h_STTC_AB[Ali, Alj]#))
        #print("STTC_AB[i, j]:   {}".format(STTC_AB[Ali, Alj]))#

        print("STTC_AB same? {}".format(np.allclose(STTC_AB, res.h_STTC_AB)))
        #print("STTC_AB:\n{}\n".format(STTC_AB))
        #print("res.h_STTC_AB:\n{}\n".format(res.h_STTC_AB))
        #print("STTC_AB:\n{}\n".format(STTC_AB[0, :50].tolist()))
        #print("res.h_STTC_AB:\n{}\n".format(res.h_STTC_AB[0, :50].tolist()))

        pass

    pass



def STTC_pairs_analysis_gpu_no_null(A, Dt, compare):
    start_time = time.time()

    N = A.shape[0]
    F = A.shape[1]

    TPB = (tpb, tpb)

    BPG_N = math.ceil(N / tpb)
    BPG_F = math.ceil(F / tpb)
    BPG_NxF = (BPG_N, BPG_F)
    BPG_Nx1 = (BPG_N, 1)
    BPG_NxN = (BPG_N, BPG_N)

    d_A = cuda.to_device(A)
    d_A_r = cuda.device_array((N, F), dtype=np.uint64)

    d_Ap = cuda.device_array((N, F), dtype=np.uint8)
    d_Ap_r = cuda.device_array((N, F), dtype=np.uint64)
    d_T_Ap = cuda.device_array(N, dtype=np.float64)

    d_B = d_A
    d_B_r = d_A_r

    d_Bm = cuda.device_array((N, F), dtype=np.uint8)
    d_Bm_r = cuda.device_array((N, F), dtype=np.uint64)
    d_T_Bm = cuda.device_array(N, dtype=np.float64)

    d_N_B_Ap = cuda.device_array((N, N), dtype=np.uint64)
    d_N_A_Bm = cuda.device_array((N, N), dtype=np.uint64)

    d_P_B_Ap = cuda.device_array((N, N), dtype=np.float64)
    d_P_A_Bm = cuda.device_array((N, N), dtype=np.float64)

    d_STTC_AB = cuda.device_array((N, N), dtype=np.float64)

    # find space occupied by spikes of neuron A in timeseries
    sum_reduce_2d[BPG_NxF, TPB](d_A, d_A_r)
    #cuda.synchronize()
    sum_reduce_2d[BPG_Nx1, TPB](d_A_r, d_A_r)
    #cuda.synchronize()

    # create tiles for neuron A
    copy_array[BPG_NxF, TPB](d_A, d_Ap)
    #cuda.synchronize()
    for t in range(1, Dt+1):
        #d_Ap[:, t:] = d_Ap[:, t:] | d_A[:, :-t]
        BPG_Fm = math.ceil((F - t) / tpb)
        BPG_NxFm = (BPG_N, BPG_Fm)
        binaryOR[BPG_NxFm, TPB](d_A[:, :-t], d_Ap[:, t:])
        #cuda.synchronize()

    # find space occupied by tiles of neuron A in timeseries
    sum_reduce_2d[BPG_NxF, TPB](d_Ap, d_Ap_r)
    #cuda.synchronize()
    sum_reduce_2d[BPG_Nx1, TPB](d_Ap_r, d_Ap_r)
    #cuda.synchronize()
    tiling_percentage[math.ceil(N / tpb_sq), tpb_sq](d_Ap_r[:, 0], F, d_T_Ap)
    #cuda.synchronize()

    # find correlation between timeseries of neuron B and tiles of neuron A
    correlation[BPG_NxN, TPB](d_B, d_Ap, d_N_B_Ap)
    #cuda.synchronize()
    correlation_percentage_x[BPG_NxN, TPB](d_N_B_Ap, d_B_r[:, 0], d_P_B_Ap)
    #cuda.synchronize()
    
    # create tiles for neuron B
    copy_array[BPG_NxF, TPB](d_B, d_Bm)
    #cuda.synchronize()
    for t in range(1, Dt+1):
        #d_Bm[:, :-t] = d_Bm[:, :-t] | d_B[:, t:]
        BPG_Fm = math.ceil((F - t) / tpb)
        BPG_NxFm = (BPG_N, BPG_Fm)
        binaryOR[BPG_NxFm, TPB](d_B[:, t:], d_Bm[:, :-t])
        #cuda.synchronize()
    
    # find space occupied by tiles of neuron B in timeseries
    sum_reduce_2d[BPG_NxF, TPB](d_Bm, d_Bm_r)
    #cuda.synchronize()
    sum_reduce_2d[BPG_Nx1, TPB](d_Bm_r, d_Bm_r)
    #cuda.synchronize()
    tiling_percentage[math.ceil(N / tpb_sq), tpb_sq](d_Bm_r[:, 0], F, d_T_Bm)
    #cuda.synchronize()

    # find correlation between timeseries of neuron A and tiles of neuron B
    correlation[BPG_NxN, TPB](d_A, d_Bm, d_N_A_Bm)
    #cuda.synchronize()
    correlation_percentage_x[BPG_NxN, TPB](d_N_A_Bm, d_A_r[:, 0], d_P_A_Bm)
    #cuda.synchronize()

    # calculate STTC
    STTC_pairs[BPG_NxN, TPB](d_T_Ap, d_P_B_Ap, d_T_Bm, d_P_A_Bm, d_STTC_AB)
    #cuda.synchronize()

    h_STTC_AB = np.empty((N, N), dtype=np.float64)
    d_STTC_AB.copy_to_host(h_STTC_AB)

    analysis_time = time.time() - start_time
    (mins, secs) = divmod(analysis_time, 60.)
    print("GPU Computational Analysis: %im%.3fs\n(+ memory transfers)" % (mins, secs))
    
    if compare == 1:
        res = from_device()
        res.enabled = 1

        h_A_r = np.empty((N, F), dtype=np.uint64)
        d_A_r.copy_to_host(h_A_r)
        res.h_A_r = h_A_r

        h_Ap = np.empty((N, F), dtype=np.uint8)
        d_Ap.copy_to_host(h_Ap)
        res.h_Ap = h_Ap

        h_Ap_r = np.empty((N, F), dtype=np.uint64)
        d_Ap_r.copy_to_host(h_Ap_r)
        res.h_Ap_r = h_Ap_r

        h_T_Ap = np.empty(N, dtype=np.float64)
        d_T_Ap.copy_to_host(h_T_Ap)
        res.h_T_Ap = h_T_Ap

        h_Bm = np.empty((N, F), dtype=np.uint8)
        d_Bm.copy_to_host(h_Bm)
        res.h_Bm = h_Bm

        h_Bm_r = np.empty((N, F), dtype=np.uint64)
        d_Bm_r.copy_to_host(h_Bm_r)
        res.h_Bm_r = h_Bm_r

        h_T_Bm = np.empty(N, dtype=np.float64)
        d_T_Bm.copy_to_host(h_T_Bm)
        res.h_T_Bm = h_T_Bm

        h_N_B_Ap = np.empty((N, N), dtype=np.uint64)
        d_N_B_Ap.copy_to_host(h_N_B_Ap)
        h_N_Ap_B = np.transpose(h_N_B_Ap)
        res.h_N_Ap_B = h_N_Ap_B

        h_P_B_Ap = np.empty((N, N), dtype=np.float64)
        d_P_B_Ap.copy_to_host(h_P_B_Ap)
        h_P_Ap_B = np.transpose(h_P_B_Ap)
        res.h_P_Ap_B = h_P_Ap_B

        h_N_A_Bm = np.empty((N, N), dtype=np.uint64)
        d_N_A_Bm.copy_to_host(h_N_A_Bm)
        res.h_N_A_Bm = h_N_A_Bm

        h_P_A_Bm = np.empty((N, N), dtype=np.float64)
        d_P_A_Bm.copy_to_host(h_P_A_Bm)
        res.h_P_A_Bm = h_P_A_Bm

        res.h_STTC_AB = h_STTC_AB

        STTC_pairs_analysis_cpu_no_null(A, Dt, res)

        pass

    pass



def STTC_triplets_analysis_gpu_rng(IDs, A, Dt, Shifts, filename):
    start_time = time.time()

    thres = .8

    N = A.shape[0]
    F = A.shape[1]
    S = 1 + Shifts
    full_filename = filename.replace('.csv', '_' + str(Shifts) + '-shifts_' + str(Dt) + '-dt_triplets.csv')

    TPB = (tpb, tpb)

    BPG_N = math.ceil(N / tpb)
    BPG_F = math.ceil(F / tpb)
    BPG_S = math.ceil(S / tpb)
    BPG_Sm = math.ceil(Shifts / tpb)
    BPG_NxF = (BPG_N, BPG_F)
    BPG_Nx1 = (BPG_N, 1)
    BPG_SxF = (BPG_S, BPG_F)
    BPG_Sx1 = (BPG_S, 1)
    BPG_SxN = (BPG_S, BPG_N)
    BPG_NxS = (BPG_N, BPG_S)
    BPG_NxSm = (BPG_N, BPG_Sm)

    d_A = cuda.to_device(A)
    d_A_r = cuda.device_array((N, F), dtype=np.uint64)

    d_C = cuda.device_array((S, F), dtype=np.uint8)
    d_Cp = cuda.device_array((S, F), dtype=np.uint8)

    d_redA = d_Cp
    d_redA_r = cuda.device_array((S, F), dtype=np.uint64)
    d_redA_rr = cuda.device_array(Shifts, dtype=np.uint64)

    d_redAp = cuda.device_array((S, F), dtype=np.uint8)
    d_redAp_r = cuda.device_array((S, F), dtype=np.uint64)
    d_T_redAp = cuda.device_array(S, dtype=np.float64)

    d_B = cuda.device_array((N, F), dtype=np.uint8)
    d_B_r = cuda.device_array((N, F), dtype=np.uint64)

    d_Bm = cuda.device_array((N, F), dtype=np.uint8)
    d_Bm_r = cuda.device_array((N, F), dtype=np.uint64)
    d_T_Bm = cuda.device_array(N, dtype=np.float64)

    d_N_Bm_redA = cuda.device_array((N, S), dtype=np.uint64)
    d_N_redAp_B = cuda.device_array((S, N), dtype=np.uint64)

    d_P_Bm_redA = cuda.device_array((N, S), dtype=np.float64)
    d_P_redAp_B = cuda.device_array((S, N), dtype=np.float64)

    d_STTC_redAB = cuda.device_array((N, S), dtype=np.float64)

    d_CG_r = cuda.device_array((N, Shifts), dtype=np.float64)
    d_CGd = cuda.device_array((N, Shifts), dtype=np.float64)
    d_CGd_r = cuda.device_array((N, Shifts), dtype=np.float64)

    d_CGmean = cuda.device_array(N, dtype=np.float64)
    d_CGstdev = cuda.device_array(N, dtype=np.float64)
    d_CGzscore = cuda.device_array(N, dtype=np.float64)

    # find space occupied by spikes of neuron A in timeseries
    sum_reduce_2d[BPG_NxF, TPB](d_A, d_A_r)
    sum_reduce_2d[BPG_Nx1, TPB](d_A_r, d_A_r)

    copy_array[BPG_NxF, TPB](d_A, d_B)
        
    # find space occupied by spikes of neuron B in timeseries
    sum_reduce_2d[BPG_NxF, TPB](d_B, d_B_r)
    sum_reduce_2d[BPG_Nx1, TPB](d_B_r, d_B_r)
        
    # create tiles for neuron B
    copy_array[BPG_NxF, TPB](d_B, d_Bm)
    for t in range(1, Dt+1):
        #d_Bm[:, :-t] = d_Bm[:, :-t] | d_B[:, t:]
        BPG_Fm = math.ceil((F - t) / tpb)
        BPG_NxFm = (BPG_N, BPG_Fm)
        binaryOR[BPG_NxFm, TPB](d_B[:, t:], d_Bm[:, :-t])
    
    # find space occupied by tiles of neuron B in timeseries
    sum_reduce_2d[BPG_NxF, TPB](d_Bm, d_Bm_r)
    sum_reduce_2d[BPG_Nx1, TPB](d_Bm_r, d_Bm_r)
    tiling_percentage[math.ceil(N / tpb_sq), tpb_sq](d_Bm_r[:, 0], F, d_T_Bm)

    # some timing prints
    some = 100 if N > 99 else 10

    some_tmp = some
    some_percent = math.ceil(N / some)
    some_sum = some_percent

    curr_time = time.time()
    some_percent_time = curr_time - start_time
    prev_time = curr_time
    (mins, secs) = divmod(some_percent_time, 60.)
    print("%i/%i - %im%.3fs" % (some - some_tmp, some, mins, secs))

    with open(full_filename, 'w') as f:
        f.write('NeuronA,NeuronB,NeuronC,STTC,CtrlGrpMean,CtrlGrpStDev,NullSTTC,Zscore\n')

    for a in range(N):

        # some timing prints
        if a == some_sum:
            some_tmp -= 1
            some_percent = math.ceil((N - some_sum) / some_tmp)
            some_sum += some_percent

            curr_time = time.time()
            some_percent_time = curr_time - prev_time
            prev_time = curr_time
            (mins, secs) = divmod(some_percent_time, 60.)
            print("%i/%i - %im%.3fs" % (some - some_tmp, some, mins, secs))

            pass
        
        for c in range(N):
            if a == c:
                continue
        
            # create the null array for neuron C
            h_shift_ary = np.random.choice(np.arange(1, F), size=S, replace=False)
            h_shift_ary[0] = 0
            d_shift_ary = cuda.to_device(h_shift_ary)
            circular_shift_rng[BPG_SxF, TPB](d_A[c, :], d_shift_ary, d_C)
            
            # create tiles for neuron C
            copy_array[BPG_SxF, TPB](d_C, d_Cp)
            for t in range(1, Dt+1):
                #d_Cp[:, t:] = d_Cp[:, t:] | d_C[:, :-t]
                BPG_Fm = math.ceil((F - t) / tpb)
                BPG_SxFm = (BPG_S, BPG_Fm)
                binaryOR[BPG_SxFm, TPB](d_C[:, :-t], d_Cp[:, t:])

            binaryAND[BPG_SxF, TPB](d_A[a, :], d_redA)

            # find number of spikes of neuron redA
            sum_reduce_2d[BPG_SxF, TPB](d_redA, d_redA_r)
            sum_reduce_2d[BPG_Sx1, TPB](d_redA_r, d_redA_r)

            #h_redA_r = d_redA_r[:, 0].copy_to_host()

            if d_redA_r[0, 0] == 0:#h_redA_r[0] == 0:
                #print("NA:%d NC:%d spikes:%d" % (IDs[a], IDs[c], h_redA_r[0]))
                continue

            binary_filter_nonzero_1d[math.ceil(Shifts / tpb_sq), tpb_sq](d_redA_r[1:, 0], d_redA_rr)
            sum_reduce_1d[math.ceil(Shifts / tpb_sq), tpb_sq](d_redA_rr, d_redA_rr)
            sum_reduce_1d[1, tpb_sq](d_redA_rr, d_redA_rr)

            #h_redA_rr = d_redA_rr[:].copy_to_host()

            if d_redA_rr[0] < int(thres*Shifts):#h_redA_rr[0] < int(thres*Shifts):
                #print("NA:%d NC:%d spikes:%d synthetic:%d" % (IDs[a], IDs[c], h_redA_r[0], h_redA_rr[0]))
                continue

            # find correlation between tiles of neuron B and timeseries of neuron redA
            correlation[BPG_NxS, TPB](d_Bm, d_redA, d_N_Bm_redA)
            correlation_percentage_y[BPG_NxS, TPB](d_N_Bm_redA, d_redA_r[:, 0], d_P_Bm_redA)

            # create tiles for neuron redA
            copy_array[BPG_SxF, TPB](d_A, d_redAp)
            for t in range(1, Dt+1):
                #d_redAp[:, t:] = d_redAp[:, t:] | d_redA[:, :-t]
                BPG_Fm = math.ceil((F - t) / tpb)
                BPG_SxFm = (BPG_S, BPG_Fm)
                binaryOR[BPG_SxFm, TPB](d_A[:, :-t], d_redAp[:, t:])

            # find space occupied by tiles of neuron redA in timeseries
            sum_reduce_2d[BPG_SxF, TPB](d_redAp, d_redAp_r)
            sum_reduce_2d[BPG_Sx1, TPB](d_redAp_r, d_redAp_r)
            tiling_percentage[math.ceil(S / tpb_sq), tpb_sq](d_redAp_r[:, 0], F, d_T_redAp)

            # find correlation between tiles of neuron redA and timeseries of neuron B
            correlation[BPG_SxN, TPB](d_redAp, d_B, d_N_redAp_B)
            correlation_percentage_y[BPG_SxN, TPB](d_N_redAp_B, d_B_r[:, 0], d_P_redAp_B)

            # calculate STTC
            STTC_triplets[BPG_NxS, TPB](d_T_redAp, d_P_redAp_B, d_T_Bm, d_P_Bm_redA, d_STTC_redAB)
        
            h_STTC_redAB = d_STTC_redAB[:, 0].copy_to_host()
            h_STTC_Null = d_STTC_redAB[:, 1].copy_to_host()

            # calculate CtrlGrpMean
            filter_nonzero_2d[BPG_NxSm, TPB](d_STTC_redAB[:, 1:], d_redAp_r[:, 0], d_STTC_redAB[:, 1:])
            sum_reduce_2d_f8[BPG_NxSm, TPB](d_STTC_redAB[:, 1:], d_CG_r)
            sum_reduce_2d_f8[BPG_Nx1, TPB](d_CG_r, d_CG_r)
            tiling_percentage[math.ceil(N / tpb_sq), tpb_sq](d_CG_r[:, 0], d_redA_rr[0], d_CGmean)
            
            h_CGmean = d_CGmean.copy_to_host()

            # calculate CtrlGrpStDev
            deviation[BPG_NxSm, TPB](d_STTC_redAB[:, 1:], d_CGmean, d_CGd)
            filter_nonzero_2d[BPG_NxSm, TPB](d_CGd, d_redAp_r[:, 0], d_CGd)
            sum_reduce_2d_f8[BPG_NxSm, TPB](d_CGd, d_CGd_r)
            sum_reduce_2d_f8[BPG_Nx1, TPB](d_CGd_r, d_CGd_r)
            sqrt_mean[math.ceil(N / tpb_sq), tpb_sq](d_CGd_r[:, 0], d_redA_rr[0], d_CGstdev)
            
            h_CGstdev = d_CGstdev.copy_to_host()

            zscore[math.ceil(N / tpb_sq), tpb_sq](d_STTC_redAB[:, 0], d_CGmean, d_CGstdev, d_CGzscore)
        
            h_CGzscore = d_CGzscore.copy_to_host()

            h_nA = np.full(N, IDs[a])

            h_nB = np.copy(IDs)

            h_nC = np.full(N, IDs[c])

            h_arr = np.column_stack((h_nA, h_nB, h_nC, h_STTC_redAB, h_CGmean, h_CGstdev, h_STTC_Null, h_CGzscore))

            h_arr = np.delete(h_arr, [a, c], 0)

            with open(full_filename, 'a') as f:
                np.savetxt(f, h_arr, fmt='%d,%d,%d,%.10f,%.10f,%.10f,%.10f,%.10f')
            
            continue

        continue

    # some timing prints
    curr_time = time.time()
    some_percent_time = curr_time - prev_time
    (mins, secs) = divmod(some_percent_time, 60.)
    print("%i/%i - %im%.3fs" % (some, some, mins, secs))
    
    pass



def STTC_scc_analysis_gpu_rng(IDs, A, Dt, Shifts, filename):
    start_time = time.time()

    N = A.shape[0]
    S = 1 + Shifts
    full_filename = filename.replace('.csv', '_' + str(Shifts) + '-shifts_' + str(Dt) + '-dt_scc.csv')

    TPB = (tpb, tpb)

    with open(full_filename, 'w') as f:
        f.write('NeuronA,NeuronB,NeuronR,STTC,CtrlGrpMean,CtrlGrpStDev,NullSTTC,Zscore\n')

    # some timing prints
    some = 100 if N > 99 else 10

    some_tmp = some
    some_percent = math.ceil(N / some)
    some_sum = some_percent

    curr_time = time.time()
    some_percent_time = curr_time - start_time
    prev_time = curr_time
    (mins, secs) = divmod(some_percent_time, 60.)
    print("%i/%i - %im%.3fs" % (some - some_tmp, some, mins, secs))

    for r in range(N):
        # some timing prints
        if r == some_sum:
            some_tmp -= 1
            some_percent = math.ceil((N - some_sum) / some_tmp)
            some_sum += some_percent

            curr_time = time.time()
            some_percent_time = curr_time - prev_time
            prev_time = curr_time
            (mins, secs) = divmod(some_percent_time, 60.)
            print("%i/%i - %im%.3fs" % (some - some_tmp, some, mins, secs))

            pass

        # remove frames where neuron R has spikes
        R = A[:, np.array(A[r, :] == 0)]

        F = R.shape[1]

        BPG_N = math.ceil(N / tpb)
        BPG_F = math.ceil(F / tpb)
        BPG_S = math.ceil(S / tpb)
        BPG_Sm = math.ceil(Shifts / tpb)
        BPG_NxF = (BPG_N, BPG_F)
        BPG_Nx1 = (BPG_N, 1)
        BPG_SxF = (BPG_S, BPG_F)
        BPG_Sx1 = (BPG_S, 1)
        BPG_SxN = (BPG_S, BPG_N)
        BPG_NxS = (BPG_N, BPG_S)
        BPG_NxSm = (BPG_N, BPG_Sm)

        d_A = cuda.to_device(R)
        d_A_r = cuda.device_array((N, F), dtype=np.uint64)

        d_Ap = cuda.device_array((N, F), dtype=np.uint8)
        d_Ap_r = cuda.device_array((N, F), dtype=np.uint64)
        d_T_Ap = cuda.device_array(N, dtype=np.float64)

        d_B = cuda.device_array((S, F), dtype=np.uint8)
        d_B_r = cuda.device_array((S, F), dtype=np.uint64)

        d_Bm = cuda.device_array((S, F), dtype=np.uint8)
        d_Bm_r = cuda.device_array((S, F), dtype=np.uint64)
        d_T_Bm = cuda.device_array(S, dtype=np.float64)

        d_N_B_Ap = cuda.device_array((S, N), dtype=np.uint64)
        d_N_A_Bm = cuda.device_array((N, S), dtype=np.uint64)

        d_P_B_Ap = cuda.device_array((S, N), dtype=np.float64)
        d_P_A_Bm = cuda.device_array((N, S), dtype=np.float64)

        d_STTC_AB = cuda.device_array((N, S), dtype=np.float64)

        d_CG_r = cuda.device_array((N, Shifts), dtype=np.float64)
        d_CGd = cuda.device_array((N, Shifts), dtype=np.float64)
        d_CGd_r = cuda.device_array((N, Shifts), dtype=np.float64)

        d_CGmean = cuda.device_array(N, dtype=np.float64)
        d_CGstdev = cuda.device_array(N, dtype=np.float64)
        d_CGzscore = cuda.device_array(N, dtype=np.float64)

        # find space occupied by spikes of neuron A in timeseries
        sum_reduce_2d[BPG_NxF, TPB](d_A, d_A_r)
        sum_reduce_2d[BPG_Nx1, TPB](d_A_r, d_A_r)

        # create tiles for neuron A
        copy_array[BPG_NxF, TPB](d_A, d_Ap)
        for t in range(1, Dt+1):
            #d_Ap[:, t:] = d_Ap[:, t:] | d_A[:, :-t]
            BPG_Fm = math.ceil((F - t) / tpb)
            BPG_NxFm = (BPG_N, BPG_Fm)
            binaryOR[BPG_NxFm, TPB](d_A[:, :-t], d_Ap[:, t:])

        # find space occupied by tiles of neuron A in timeseries
        sum_reduce_2d[BPG_NxF, TPB](d_Ap, d_Ap_r)
        sum_reduce_2d[BPG_Nx1, TPB](d_Ap_r, d_Ap_r)
        tiling_percentage[math.ceil(N / tpb_sq), tpb_sq](d_Ap_r[:, 0], F, d_T_Ap)

        # for each neuron in dataset
        for n in range(N):
            # create the null array for neuron B
            h_shift_ary = np.random.randint(1, high=F, size=S)
            h_shift_ary[0] = 0
            d_shift_ary = cuda.to_device(h_shift_ary)
            circular_shift_rng[BPG_SxF, TPB](d_A[n, :], d_shift_ary, d_B)

            # find space occupied by spikes of neuron B in timeseries
            sum_reduce_2d[BPG_SxF, TPB](d_B, d_B_r)
            sum_reduce_2d[BPG_Sx1, TPB](d_B_r, d_B_r)

            # find correlation between timeseries of neuron B and tiles of neuron A
            correlation[BPG_SxN, TPB](d_B, d_Ap, d_N_B_Ap)
            correlation_percentage_x[BPG_SxN, TPB](d_N_B_Ap, d_B_r[:, 0], d_P_B_Ap)

            # create tiles for neuron B
            copy_array[BPG_SxF, TPB](d_B, d_Bm)
            for t in range(1, Dt+1):
                #d_Bm[:, :-t] = d_Bm[:, :-t] | d_B[:, t:]
                BPG_Fm = math.ceil((F - t) / tpb)
                BPG_SxFm = (BPG_S, BPG_Fm)
                binaryOR[BPG_SxFm, TPB](d_B[:, t:], d_Bm[:, :-t])

            # find space occupied by tiles of neuron B in timeseries
            sum_reduce_2d[BPG_SxF, TPB](d_Bm, d_Bm_r)
            sum_reduce_2d[BPG_Sx1, TPB](d_Bm_r, d_Bm_r)
            tiling_percentage[math.ceil(S / tpb_sq), tpb_sq](d_Bm_r[:, 0], F, d_T_Bm)

            # find correlation between timeseries of neuron A and tiles of neuron B
            correlation[BPG_NxS, TPB](d_A, d_Bm, d_N_A_Bm)
            correlation_percentage_x[BPG_NxS, TPB](d_N_A_Bm, d_A_r[:, 0], d_P_A_Bm)

            # calculate STTC
            STTC_pairs[BPG_NxS, TPB](d_T_Ap, d_P_B_Ap, d_T_Bm, d_P_A_Bm, d_STTC_AB)

            h_STTC_AB = d_STTC_AB[:, 0].copy_to_host()
            h_STTC_Null = d_STTC_AB[:, 1].copy_to_host()

            # calculate CtrlGrpMean
            sum_reduce_2d_f8[BPG_NxSm, TPB](d_STTC_AB[:, 1:], d_CG_r)
            sum_reduce_2d_f8[BPG_Nx1, TPB](d_CG_r, d_CG_r)
            tiling_percentage[math.ceil(N / tpb_sq), tpb_sq](d_CG_r[:, 0], Shifts, d_CGmean)

            h_CGmean = d_CGmean.copy_to_host()

            # calculate CtrlGrpStDev
            deviation[BPG_NxSm, TPB](d_STTC_AB[:, 1:], d_CGmean, d_CGd)
            sum_reduce_2d_f8[BPG_NxSm, TPB](d_CGd, d_CGd_r)
            sum_reduce_2d_f8[BPG_Nx1, TPB](d_CGd_r, d_CGd_r)
            sqrt_mean[math.ceil(N / tpb_sq), tpb_sq](d_CGd_r[:, 0], Shifts, d_CGstdev)

            h_CGstdev = d_CGstdev.copy_to_host()

            zscore[math.ceil(N / tpb_sq), tpb_sq](d_STTC_AB[:, 0], d_CGmean, d_CGstdev, d_CGzscore)
 
            h_CGzscore = d_CGzscore.copy_to_host()

            h_nA = np.copy(IDs)

            h_nB = np.full(N, IDs[n])

            h_nR = np.full(N, IDs[r])

            h_arr = np.column_stack((h_nA, h_nB, h_nR, h_STTC_AB, h_CGmean, h_CGstdev, h_STTC_Null, h_CGzscore))

            h_arr = np.delete(h_arr, n, 0)

            with open(full_filename, 'a') as f:
                np.savetxt(f, h_arr, fmt='%d,%d,%d,%.10f,%.10f,%.10f,%.10f,%.10f')

            continue

        continue

    # some timing prints
    curr_time = time.time()
    some_percent_time = curr_time - prev_time
    (mins, secs) = divmod(some_percent_time, 60.)
    print("%i/%i - %im%.3fs" % (some, some, mins, secs))

    pass



def STTC_triplets_stats(IDs, A, Dt, filename):
    start_time = time.time()

    N = A.shape[0]
    F = A.shape[1]
    full_filename = filename.replace('.csv', '_stats.csv')

    TPB = (tpb, tpb)

    BPG_N = math.ceil(N / tpb)
    BPG_F = math.ceil(F / tpb)
    BPG_NxF = (BPG_N, BPG_F)
    BPG_Nx1 = (BPG_N, 1)

    d_A = cuda.to_device(A)
    d_A_r = cuda.device_array((N, F), dtype=np.uint64)
    d_T_A = cuda.device_array(N, dtype=np.float64)

    d_Ap = cuda.device_array((N, F), dtype=np.uint8)

    d_redA = cuda.device_array((N, F), dtype=np.uint8)
    d_redA_r = cuda.device_array((N, F), dtype=np.uint64)
    d_T_redA = cuda.device_array(N, dtype=np.float64)

    # find space occupied by spikes of neuron A in timeseries
    sum_reduce_2d[BPG_NxF, TPB](d_A, d_A_r)
    sum_reduce_2d[BPG_Nx1, TPB](d_A_r, d_A_r)
    tiling_percentage[math.ceil(N / tpb_sq), tpb_sq](d_A_r[:, 0], F*0.158, d_T_A)

    h_T_A = d_T_A.copy_to_host()

    '''
    A_r = np.sum(A, dtype = np.uint64, axis=1)
    T_A = A_r / np.float64(F*0.158)

    equality = h_T_A == T_A

    activity = h_T_A >= 0.01

    print("T_A same? {}".format(np.array_equal(h_T_A, T_A)))
    '''

    print("neurons nonzero %i" % (np.count_nonzero(h_T_A > 0.0)))
    print("neurons active %i" % (np.count_nonzero(h_T_A >= 0.01)))
    print("neurons min activity %.3f" % (h_T_A.min()))
    print("neurons max activity %.3f" % (h_T_A.max()))

    '''
    with open(full_filename, 'w') as f:
        f.write('Neuron,d_activity,h_activity,equal,>=0.01\n')

    arr = np.column_stack((IDs, h_T_A, T_A, equality, activity))

    with open(full_filename, 'a') as f:
        np.savetxt(f, arr, fmt='%d,%.4f,%.4f,%d,%d')
    '''

    # create tiles for neuron A
    copy_array[BPG_NxF, TPB](d_A, d_Ap)
    for t in range(1, Dt+1):
        #d_Ap[:, t:] = d_Ap[:, t:] | d_A[:, :-t]
        BPG_Fm = math.ceil((F - t) / tpb)
        BPG_NxFm = (BPG_N, BPG_Fm)
        binaryOR[BPG_NxFm, TPB](d_A[:, :-t], d_Ap[:, t:])
    
    # some timing prints
    some = 10#0 if N > 999 else 10
    some_percent = int(N / some)
    prev_time = start_time

    tr_nonzero = 0
    tr_active = 0

    tr_min = 0.5
    tr_max = 0.0
    tr_mean = 0.0

    for a in range(N):

        # some timing prints
        if (a % some_percent) == 0:
            curr_time = time.time()
            some_percent_time = curr_time - prev_time
            prev_time = curr_time
            (mins, secs) = divmod(some_percent_time, 60.)
            print("%i/%i - %im%.3fs" % (a // some_percent, some, mins, secs))
            pass

        copy_array[BPG_NxF, TPB](d_Ap, d_redA)

        binaryAND[BPG_NxF, TPB](d_A[a, :], d_redA)

        # find space occupied by spikes of neuron A in timeseries
        sum_reduce_2d[BPG_NxF, TPB](d_redA, d_redA_r)
        sum_reduce_2d[BPG_Nx1, TPB](d_redA_r, d_redA_r)
        tiling_percentage[math.ceil(N / tpb_sq), tpb_sq](d_redA_r[:, 0], F*0.158, d_T_redA)

        h_T_redA = d_T_redA.copy_to_host()

        T_redA = np.delete(h_T_redA, a)

        #print("neuron %i: min %.3f max %.3f" % (a, h_T_redA.min(), h_T_redA.max()))

        tr_nonzero += np.count_nonzero(T_redA > 0.0)
        tr_active += np.count_nonzero(T_redA >= 0.01)

        tr_min = tr_min if tr_min < T_redA.min() else T_redA.min()
        tr_max = tr_max if tr_max > T_redA.max() else T_redA.max()
        tr_mean += T_redA.sum()

        continue

    tr_ttl = N*(N-1)

    print("reduced A nonzero %i (%.3f%%)" % (tr_nonzero, tr_nonzero * 100 / tr_ttl))
    print("reduced A active %i (%.3f%%)" % (tr_active, tr_active * 100 / tr_ttl))

    print("reduced A min %.3f" % (tr_min))
    print("reduced A max %.3f" % (tr_max))
    print("reduced A mean %.3f" % (tr_mean / tr_ttl))

    pass



def STTC_triplets_ids_rng(IDs, A, Dt, Shifts, filename):
    start_time = time.time()

    thres = .8

    N = A.shape[0]
    F = A.shape[1]
    S = 1 + Shifts
    full_filename = filename.replace('.csv', '_' + str(Shifts) + '-shifts_' + str(Dt) + '-dt_triplets_ids.csv')

    TPB = (tpb, tpb)

    BPG_N = math.ceil(N / tpb)
    BPG_F = math.ceil(F / tpb)
    BPG_S = math.ceil(S / tpb)
    BPG_Sm = math.ceil(Shifts / tpb)
    BPG_NxF = (BPG_N, BPG_F)
    BPG_Nx1 = (BPG_N, 1)
    BPG_SxF = (BPG_S, BPG_F)
    BPG_Sx1 = (BPG_S, 1)
    BPG_SxN = (BPG_S, BPG_N)
    BPG_NxS = (BPG_N, BPG_S)
    BPG_NxSm = (BPG_N, BPG_Sm)

    d_A = cuda.to_device(A)
    d_A_r = cuda.device_array((N, F), dtype=np.uint64)

    d_C = cuda.device_array((S, F), dtype=np.uint8)
    d_Cp = cuda.device_array((S, F), dtype=np.uint8)

    d_redA = d_Cp
    d_redA_r = cuda.device_array((S, F), dtype=np.uint64)
    d_redA_rr = cuda.device_array(Shifts, dtype=np.uint64)

    # some timing prints
    some = 100 if N > 999 else 10
    some_percent = math.ceil(N / some)
    prev_time = start_time

    with open(full_filename, 'w') as f:
        f.write('NeuronA,NeuronB,NeuronC\n')

    for a in range(N):

        # some timing prints
        if (a % some_percent) == 0:
            curr_time = time.time()
            some_percent_time = curr_time - prev_time
            prev_time = curr_time
            (mins, secs) = divmod(some_percent_time, 60.)
            print("%i/%i - %im%.3fs" % (a // some_percent, some, mins, secs))
            pass
        
        for c in range(N):
            if a == c:
                continue
        
            # create the null array for neuron C
            h_shift_ary = np.random.randint(1, high=F, size=S)
            h_shift_ary[0] = 0
            d_shift_ary = cuda.to_device(h_shift_ary)
            circular_shift_rng[BPG_SxF, TPB](d_A[c, :], d_shift_ary, d_C)
            
            # create tiles for neuron C
            copy_array[BPG_SxF, TPB](d_C, d_Cp)
            for t in range(1, Dt+1):
                #d_Cp[:, t:] = d_Cp[:, t:] | d_C[:, :-t]
                BPG_Fm = math.ceil((F - t) / tpb)
                BPG_SxFm = (BPG_S, BPG_Fm)
                binaryOR[BPG_SxFm, TPB](d_C[:, :-t], d_Cp[:, t:])

            binaryAND[BPG_SxF, TPB](d_A[a, :], d_redA)

            # find number of spikes of neuron redA
            sum_reduce_2d[BPG_SxF, TPB](d_redA, d_redA_r)
            sum_reduce_2d[BPG_Sx1, TPB](d_redA_r, d_redA_r)

            h_redA_r = d_redA_r[0:1, 0].copy_to_host()

            if h_redA_r[0] == 0:
                #print("NA:%d NC:%d spikes:%d" % (IDs[a], IDs[c], h_redA_r[0]))
                continue

            binary_filter_nonzero_1d[math.ceil(Shifts / tpb_sq), tpb_sq](d_redA_r[1:, 0], d_redA_rr)
            sum_reduce_1d[math.ceil(Shifts / tpb_sq), tpb_sq](d_redA_rr, d_redA_rr)
            sum_reduce_1d[1, tpb_sq](d_redA_rr, d_redA_rr)

            h_redA_rr = d_redA_rr[0:1].copy_to_host()

            if h_redA_rr[0] < int(thres*Shifts):
                #print("NA:%d NC:%d spikes:%d synthetic:%d" % (IDs[a], IDs[c], h_redA_r[0], h_redA_rr[0]))
                continue

            h_nA = np.full(N, IDs[a])

            h_nB = np.copy(IDs)

            h_nC = np.full(N, IDs[c])

            h_arr = np.column_stack((h_nA, h_nB, h_nC))

            h_arr = np.delete(h_arr, [a, c], 0)

            with open(full_filename, 'a') as f:
                np.savetxt(f, h_arr, fmt='%d,%d,%d')
            
            continue

        continue

    # some timing prints
    curr_time = time.time()
    some_percent_time = curr_time - prev_time
    prev_time = curr_time
    (mins, secs) = divmod(some_percent_time, 60.)
    print("%i/%i - %im%.3fs" % (some, some, mins, secs))
    
    pass



def main():
    filename = sys.argv[1]
    print(filename)

    #For running multiple .csv datasets that their path is written in the input .txt file
    if filename[-4:] == '.txt':
        Dt = int(sys.argv[2])
        N_shifts = int(sys.argv[3])
        cuda.select_device(int(sys.argv[4]))
        
        txtfile = open(filename)
        files = np.loadtxt(txtfile, dtype=str)

        #dir = filename.replace(filename.rpartition('/')[-1], '')
        dir = str(Path(filename).parent) + '/'
        print(dir)

        for i in range(1):#files.shape[0]):
            data_path = dir + files[i]
            with open(data_path, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                header = next(reader)
                dataset = np.array(list(reader)).astype(np.uint8)
            
            ids = [int(c.replace('V', '')) for c in header]
            
            dataset = dataset.T
            
            print()
            
            print(dataset.shape)
            
            print(data_path)
            
            res_path = dir + 'results/' + files[i]
            
            Path(dir + 'results/').mkdir(parents=True, exist_ok=True)
            
            STTC_pairs_analysis_gpu_rng(ids, dataset, Dt, N_shifts, res_path)
            
            continue

        pass
    #For running a single .csv dataset
    elif filename[-4:] == '.csv':
        Dt = int(sys.argv[2])
        N_shifts = int(sys.argv[3])
        cuda.select_device(int(sys.argv[4]))
        
        dir = str(Path(filename).parent) + '/'

        #csvfile = open(filename)
        #dataset = np.loadtxt(csvfile, delimiter=',', dtype=np.uint8, skiprows=skip)
        data_path = filename
        with open(data_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)
            dataset = np.array(list(reader)).astype(np.uint8)
        
        ids = [int(c.replace('V', '')) for c in header]
        #ids = np.ascontiguousarray(ids[0:100])

        dataset = dataset.T
        #dataset = np.ascontiguousarray(dataset[0:100, :])

        '''
        N_neurons = int(sys.argv[5])
        N_frames = int(sys.argv[6])
        if N_neurons != dataset.shape[0] or N_frames != dataset.shape[1]:
            print(dataset.shape)
            sys.exit('Error: Check dataset size. Format should be (Neurons x Frames).')
            pass
        '''

        print()

        #dataset = dataset[:, 0:int(N_frames*sixth/6)]
        print(dataset.shape)

        #data_path = data_path.replace('_full_60min', '_1st-' + str(sixth*10) + 'min')
        #data_path = data_path.replace('.csv', '')
        print(data_path)
        
        res_path = dir + 'results/' + filename
        
        Path(dir + 'results/').mkdir(parents=True, exist_ok=True)
        
        '''
        #STTC pairs analysis in CPU without control group
        STTC_pairs_analysis_cpu_no_null(dataset, Dt, from_device())
        '''
        '''
        #STTC pairs analysis in GPU without control group
        STTC_pairs_analysis_gpu_no_null(dataset, Dt, compare=1)
        '''
        '''
        #STTC pairs analysis in GPU with all shifts for control group
        STTC_pairs_analysis_gpu(ids, dataset, Dt, res_path)
        '''
        '''
        #STTC pairs analysis in GPU with specific number of shifts for control group
        STTC_pairs_analysis_gpu_rng(ids, dataset, Dt, N_shifts, res_path)
        '''
        '''
        #STTC triplets analysis in GPU with specific number of shifts for control group
        STTC_triplets_analysis_gpu_rng(ids, dataset, Dt, N_shifts, res_path)
        '''
        #STTC strict clustering coeff analysis in GPU with specific number of shifts for control group
        STTC_scc_analysis_gpu_rng(ids, dataset, Dt, N_shifts, res_path)
        '''
        STTC_triplets_stats(ids, dataset, Dt, res_path)
        '''
        '''
        STTC_triplets_ids_rng(ids, dataset, Dt, N_shifts, res_path)
        '''

        pass
    else:
        sys.exit('Error: Not supported type of file.')

    pass

if __name__=='__main__':
    main()
