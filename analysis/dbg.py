import numpy as np, json, os, sys
sys.argv=['x']
src=open('analysis/fit_proj.py').read().split('T=json.load')[0]
g={}
exec(src, g)
p=np.r_[np.array(json.load(open('analysis/transform_final.json'))['B']).ravel(),
        np.array(json.load(open('analysis/transform_final.json'))['c'])]
for proj in (False,True):
    pp=np.r_[p,[0.0,0.0]] if proj else p
    f=g['make_loss'](proj)
    v=f(pp)
    print("proj",proj,"type",type(v),np.shape(v),v)
    d=g['parts'](pp,proj)
    for k in ('lq','gq','s','rho','lnd','box'):
        print("   ",k,type(d[k]),np.shape(d[k]))
