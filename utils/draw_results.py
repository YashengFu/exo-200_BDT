import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import numpy as np

def prepPredArr(Ytr, Yts, Ypredtr, Ypredts):
    YpredSgts=[]
    YpredSgtr=[]
    YpredBgts=[]
    YpredBgtr=[]
    for i in range(len(Ytr)):
        if Ytr[i]==1:
            YpredSgtr.append(Ypredtr[i])
        elif Ytr[i]==0:
            YpredBgtr.append(Ypredtr[i])

    for i in range(len(Yts)):
        if Yts[i]==1:
            YpredSgts.append(Ypredts[i])
        elif Yts[i]==0:
            YpredBgts.append(Ypredts[i])

    YpredSgts=np.array(YpredSgts)
    YpredSgtr=np.array(YpredSgtr)
    YpredBgts=np.array(YpredBgts)
    YpredBgtr=np.array(YpredBgtr)
    return YpredSgts,YpredSgtr,YpredBgts,YpredBgtr

def getROCaxes(YpredSgts, YpredSgtr, YpredBgts, YpredBgtr, nbins=np.arange(0,1,0.01)):
    n_sgts, bins_sgts, _sgts = plt.hist(YpredSgts, bins=nbins, weights=np.repeat(1./len(YpredSgts), len(YpredSgts)))
    n_sgtr, bins_sgtr, _sgtr = plt.hist(YpredSgtr, bins=nbins, weights=np.repeat(1./len(YpredSgtr), len(YpredSgtr)))
    n_bgts, bins_bgts, _bgts = plt.hist(YpredBgts, bins=nbins, weights=np.repeat(1./len(YpredBgts), len(YpredBgts)))
    n_bgtr, bins_bgtr, _bgtr = plt.hist(YpredBgtr, bins=nbins, weights=np.repeat(1./len(YpredBgtr), len(YpredBgtr)))

    nSgEffTr=[]
    nSgEffTs=[]
    nBgRejTr=[]
    nBgRejTs=[]
    nbins = 100

    for i in range(nbins):
        nSgEffTr.append(sum(n_sgtr[i:nbins])/sum(n_sgtr[0:nbins]))
        nSgEffTs.append(sum(n_bgtr[0:i])/sum(n_bgtr[0:nbins]))
        nBgRejTr.append(sum(n_sgts[i:nbins])/sum(n_sgts[0:nbins]))
        nBgRejTs.append(sum(n_bgts[0:i])/sum(n_bgts[0:nbins]))

    nSgEffTr = np.array(nSgEffTr)
    nSgEffTs = np.array(nSgEffTs)
    nBgRejTr = np.array(nBgRejTr)
    nBgRejTs = np.array(nBgRejTs)
   
    return nSgEffTr,nSgEffTs,nBgRejTr,nBgRejTs

def drawOut(YpredSgts, YpredSgtr, YpredBgts, YpredBgtr, ftsz=10, nbins=np.arange(0,1,0.02),ms_value=3):
    plt.figure(figsize=(8, 6))
    plt.hist(YpredSgtr, bins=nbins, weights=np.repeat(1./len(YpredSgtr), len(YpredSgtr)), histtype='step',color="red",label='Signal Train')
    PredSts_data,PredSts_bins,patches = plt.hist(YpredSgts, bins=nbins, weights=np.repeat(1./len(YpredSgts), len(YpredSgts)), \
         histtype='stepfilled',color="red",alpha=0)
    bin_centers = 0.5*(PredSts_bins[1:] + PredSts_bins[:-1])
    plt.errorbar(bin_centers,PredSts_data,xerr=0.01,fmt='o',ecolor='black',mfc='red',mec='red',ms=ms_value,capsize=0,label='Signal Test')

    plt.hist(YpredBgtr, bins=nbins, weights=np.repeat(1./len(YpredBgtr), len(YpredBgtr)), histtype='step',color="blue",label='Background Train')
    PredBts_data,PredBts_bins,patches = plt.hist(YpredBgts, bins=nbins, weights=np.repeat(1./len(YpredBgts), len(YpredBgts)), histtype='stepfilled', color="blue", alpha=0)
    bin_centers = 0.5*(PredBts_bins[1:] + PredBts_bins[:-1])
    plt.errorbar(bin_centers,PredBts_data,xerr=0.01,fmt='o',ecolor='black',mfc='blue',mec='blue',ms=ms_value,capsize=0,label='Background Test')

    plt.title('Overtraining Test - BDT', fontsize=ftsz)
    plt.ylabel('Density', fontsize=ftsz)
    plt.xlabel('BDT Prediction', fontsize=ftsz)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend( fontsize=ftsz, loc='upper right')
    plt.savefig("./pdfs/BDT_overfit_testing.pdf")

def auc(xp, fp):
    xp=xp[::-1]
    fp=fp[::-1]
    nsamples = 1000
    step = 1./nsamples
    x = np.arange(0, 1, step) + step/2.
    area = 0.
    for xi in x:
        area += step*(np.interp(xi,xp,fp))
    return area

def draw_ROC(SgEffTr_BDT,SgEffTs_BDT,BgRejTr_BDT,BgRejTs_BDT,ftsz=10):
    test_auc = auc(SgEffTs_BDT, BgRejTs_BDT)
    train_auc = auc(SgEffTr_BDT, BgRejTr_BDT)
    plt.figure(figsize=(8, 6))
    plt.plot(SgEffTr_BDT, BgRejTr_BDT, color='blue', linestyle='-.', lw=2, label=('Training Sample, AUC %0.4f'%train_auc))
    plt.plot(SgEffTs_BDT, BgRejTs_BDT, color='red', linestyle=':', lw=2, label=('Testing Sample, AUC %0.4f'%test_auc))

    plt.title(r'$2\nu\beta\beta$ decay of $^{136}$Xe to the excited state of $^{136}$Ba', fontsize=ftsz)
    plt.ylabel('Background Rejection', fontsize=ftsz)
    plt.xlabel('Signal Efficiency', fontsize=ftsz)
    plt.legend(loc='best', fontsize=ftsz)
    plt.xlim([0.5, 1.0])
    plt.ylim([0.5, 1.0])
    plt.xticks(fontsize=ftsz)
    plt.yticks(fontsize=ftsz)
    plt.grid(True)
    plt.savefig('./pdfs/BDT_roc.pdf')