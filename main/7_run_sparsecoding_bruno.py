# %% [markdown]
# # https://github.com/yubeic/Sparse-Coding/

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy.linalg as la
import scipy.io
import PIL
import sys
from importlib import reload
import sparsify_PyTorch
# import utility
# from pytictoc import TicToc
import torch
# t=TicToc()
import sys
import os
# t.tic()
#%%
dataload_path='./data/bruno_sparse/IMAGES_Vanhateren.npy' #shape (512,512,35)
mainsavepath=f'./Results/7_bruno_sparsecoding/' #save path for the basis
os.makedirs(mainsavepath,exist_ok=True)
xdim = 16 #Patch size
ydim = 16 #Patch size

for BASIS1_NUM in [256,1024,2048]:
    args=xdim,ydim,BASIS1_NUM
    print('xdim, ydim,BASIS1_NUM ',xdim,ydim,BASIS1_NUM)
    savename=f"{mainsavepath}/Bruno_BASIS1_NUM_{BASIS1_NUM}_size{xdim}.npy"
    vm=np.load(dataload_path)#shape (512,512,40)
    print('vm',vm.shape)
    # fig = plt.figure(figsize=(10,10))
    # ax = fig.gca()
    # utility.imshow(vm[:,:,34],ax=ax)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device',device)
    # torch.cuda.set_device(1) #use GPU 1
    # Now let's start to learn sparse coding basis
    # Effective dimensionality is about 200, let's make it 20 times overcomplete.
    # Layer1 sparse coding initialization
    # xdim = 9 #Patch size
    # ydim = 9 #Patch size
    # BASIS1_NUM = 128
    xdim,ydim,BASIS1_NUM=args
    # BASIS1_NUM=100
    BASIS1_SIZE = [xdim*ydim, BASIS1_NUM]
    BATCH_SIZE = 100

    basis1 = torch.randn(BASIS1_SIZE).to(device)
    basis1 = basis1.div_(basis1.norm(2,0)) #torch.Size([256, 2048])

    lambd = 1.0
    ACT_HISTORY_LEN = 300
    HessianDiag = torch.zeros(BASIS1_NUM).to(device)
    ActL1 = torch.zeros(BASIS1_NUM).to(device)
    signalEnergy = 0.
    noiseEnergy = 0.

    edgeBuff = 5
    spRange_x = vm.shape[0] - xdim - edgeBuff * 2
    spRange_y = vm.shape[1] - ydim - edgeBuff * 2
    spRange_t = vm.shape[2]
    I = np.zeros([xdim*ydim,BATCH_SIZE]).astype('float32')
    totalSteps1 = 0
    basis1 = torch.randn(BASIS1_SIZE).to(device)
    basis1 = basis1.div_(basis1.norm(2,0))
    # t.tic()
    # STEPS=50000
    STEPS = 518801  #todo,518801 take two days for 2048, (16,16) on GPU;  
    for i in range(totalSteps1,STEPS):
        for j in range(BATCH_SIZE):
            xIdx = np.floor(np.random.rand()*spRange_x + edgeBuff).astype(int)
            yIdx = np.floor(np.random.rand()*spRange_y + edgeBuff).astype(int)
            sIdx = np.floor(np.random.rand()*spRange_t).astype(int)
            I[:,j] = vm[xIdx:xIdx+xdim,yIdx:yIdx+ydim,sIdx].reshape([xdim*ydim])
        I_cuda = torch.from_numpy(I).to(device)
        #Sparse Coefficients Inference by ISTA
        #For positive-only codes, use ISTA
        #For positive-negative codes, use ISTA_PN 
        ahat, Res = sparsify_PyTorch.ISTA_PN(I=I_cuda,basis=basis1,lambd=0.08,num_iter=1000,device=device)
        #ahat, Res = sparsify_PyTorch.ISTA(I_cuda, basis1, 0.03, 1000)

        #Statistics Collection
        ActL1 = ActL1.mul((ACT_HISTORY_LEN-1.0)/ACT_HISTORY_LEN) + ahat.abs().mean(1)/ACT_HISTORY_LEN
        HessianDiag = HessianDiag.mul((ACT_HISTORY_LEN-1.0)/ACT_HISTORY_LEN) + torch.pow(ahat,2).mean(1)/ACT_HISTORY_LEN

        signalEnergy = signalEnergy*((ACT_HISTORY_LEN-1.0)/ACT_HISTORY_LEN) + torch.pow(I_cuda,2).sum()/ACT_HISTORY_LEN
        noiseEnergy = noiseEnergy*((ACT_HISTORY_LEN-1.0)/ACT_HISTORY_LEN) + torch.pow(Res,2).sum()/ACT_HISTORY_LEN
        snr = signalEnergy/noiseEnergy

        #Dictionary Update
        totalSteps1 = totalSteps1 + 1
        basis1 = sparsify_PyTorch.quadraticBasisUpdate(basis1, Res, ahat, 0.001, HessianDiag, 0.005)
        #Print Information
        if i % 10000 == 0:
            # print('totalsteps',totalSteps1, snr, HessianDiag.min(), HessianDiag.max(), ActL1.min(), ActL1.max(), ActL1.sum())
            print('totalsteps',totalSteps1)
        basis1_host = basis1.cpu().numpy()
        print(basis1_host.shape)
        if os.path.exists(savename):
            print(f"file {savename} exists, skip saving")
        else:
            np.save(savename,basis1_host)
    print('Finish...')
    print(args)
    print('finish all', 'savename:',savename)
    # t.toc()
#%%
for BASIS1_NUM in [256, 1024, 2048]:
    patches=np.load(f'./Results/7_bruno_sparsecoding/Bruno_BASIS1_NUM_{BASIS1_NUM}_size16.npy').reshape(16,16,BASIS1_NUM)

    fig, axes = plt.subplots(16, 16, figsize=(16, 16))
    for i, ax in enumerate(axes.flat):
        ax.imshow(patches[:, :, i], cmap="gray")
        ax.axis("off")
    plt.tight_layout(pad=0.1)

    plt.savefig(f'./Results/7_bruno_sparsecoding/Bruno_BASIS1_NUM_{BASIS1_NUM}_size16_grid.png', dpi=300, bbox_inches="tight")
    plt.show()