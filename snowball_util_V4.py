# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 22:30:05 2021

@author: liuwei
"""

import pandas as pd
import numpy as np
from math import *
from random import seed,uniform
from copy import deepcopy
from scipy.linalg import lu
from numba import jit
import operator

#%% 用pde和MC解法求场外期权价格
# 场外期权类
class OTC_option(object):
    def __init__(self,duration,coupon,strike=100,multiplier=100,\
                 CDS_swap=0.0,r=0.03,rp=0.0,q=0.0):
        self._strike = strike
        self._multiplier = multiplier
        self._duration = duration # 实际交易天数
        self._CDS_swap = CDS_swap
        self._r = r
        self._coupon = coupon # coupon为年化收益率
        self._rp = rp
        self._q = q
    def set_und(self,S0,und,vol):
        self._S0 = S0    # 合同里签的价格
        self._und = und
        self._vol = vol  # 标的资产波动率

# 各种装饰器
# 设定向上敲出障碍
def up_ko_barrier(cls):
    def _set_up_ko_barrier(self,barriers,obs_dts,obs_type):
        self._up_ko_type = obs_type
        self._up_ko_barriers = barriers
        self._up_ko_obs_dts = obs_dts
    cls.set_up_ko_barrier = _set_up_ko_barrier
    return cls

# 设定向下敲入障碍
def down_ki_barrier(cls):
    def _set_down_ki_barrier(self,barriers,obs_dts,obs_type):
        self._down_ki_type = obs_type
        self._down_ki_barriers = barriers
        self._down_ki_obs_dts = obs_dts
    cls.set_down_ki_barrier = _set_down_ki_barrier
    return cls

# 是否可用pde求解
def pde_solvable(cls):
    def _set_mesh(self,S_max,S_min=0,M=100,N=None,set_inside=False):
        days_per_year = self._days_per_year
        r = self._r
        q = self._q
        sigma = self._vol
        p = self._coupon
        CDS = self._CDS_swap
        S = self._S0
        T = self._duration/days_per_year
        if N == None:
            N = self._duration
        
        self._dx = (S_max-S_min)/M
        self._xs = xs = -np.arange(0,M+1)*self._dx+S_max
        self._dt = T/N
        self._ts = ts = np.arange(0,N+1)*self._dt
        self._M = M
        self._N = N
        self._mesh = pd.DataFrame(np.zeros(shape=(M+1,N+1))*np.nan,index=xs,columns=ts)
        
        if '_up_bound_cond' in self.__dict__.keys():
            self._mesh.iloc[0,:] = self._up_bound_cond(self)
        if '_down_bound_cond' in self.__dict__.keys():
            self._mesh.iloc[-1,:] = self._down_bound_cond(self)
        if '_final_cond' in self.__dict__.keys():
            self._mesh.iloc[:,-1] = self._final_cond(self)
        if '_inside_conds' in self.__dict__.keys() and set_inside:
            for cond in self._inside_conds:
                # cond:[横坐标集，纵坐标集，函数]
                self._mesh.iloc[cond[0],cond[1]] = cond[2](self,cond[0],cond[1])
    
    def _solve_pde(self,method='Crank-Nicholson'):
        sigma = self._vol
        r = self._r
        q = self._q
        CDS = self._CDS_swap
        ps = self._xs
        ts = self._ts
        dt = self._dt
        dx = self._dx
        if method=='Crank-Nicholson':
            ps_tmp = np.array(self._mesh.iloc[1:-1,:].index)
            max_idx = self._mesh.iloc[1:-1,:].index.max()
            min_idx = self._mesh.iloc[1:-1,:].index.min()
            
            aj = np.array([-0.25*(np.power(sigma/dx,2)-(r-q-0.5*sigma*sigma)/dx)]*len(ps_tmp))
            bj = np.array([(1/dt+(r+CDS)/2+np.power(sigma/dx,2)/2)]*len(ps_tmp))
            cj = np.array([-0.25*(np.power(sigma/dx,2)+(r-q-0.5*sigma*sigma)/dx)]*len(ps_tmp))
            
            A = np.diag(aj,1)[:-1,:-1]+np.diag(bj)+np.diag(cj,-1)[1:,1:]
            A_inv = np.linalg.inv(A)
            
            aj_p = -aj
            bj_p = bj-(r+CDS)-np.power(sigma/dx,2)
            cj_p = -cj
            
            A_p = np.diag(aj_p,1)[:-1,:-1]+np.diag(bj_p)+np.diag(cj_p,-1)[1:,1:]
            
            np_mesh = self._mesh.values.astype(float)
            
            for j in range(1,np_mesh.shape[1]):
                idx4fill = np.isnan(np_mesh[1:-1,-j-1])
                
                pre_vs = np.ascontiguousarray(np_mesh[1:-1,-j])
                
                up_bound_v = np_mesh[0,-j-1]
                down_bound_v = np_mesh[-1,-j-1]
                up_bound_pre_v = np_mesh[0,-j]
                down_bound_pre_v = np_mesh[-1,-j]
                
                B = np.dot(A_p,pre_vs)+np.array([0]*(pre_vs.shape[0]-1)+[-aj[-1]*down_bound_v+aj_p[-1]*down_bound_pre_v])+\
                    np.array([-cj[0]*up_bound_v+cj_p[0]*up_bound_pre_v]+[0]*(pre_vs.shape[0]-1))
                vs = np.dot(A_inv,B)
                np_mesh[1:-1][idx4fill,-j-1] = vs[idx4fill]
            self._mesh = pd.DataFrame(np_mesh,index=self._mesh.index,columns=self._mesh.columns)
    cls.set_mesh = _set_mesh
    cls.solve_pde = _solve_pde
    return cls

# 是否可用MC求解
def mc_solvable(cls):
    def _set_mcpaths(self,seed_n=999,n_paths=100000):
        seed(seed_n)
        self._paths = mc_paths(self._S0,self._vol,self._duration,r=self._r,M=n_paths,days_per_year=self._days_per_year,q=self._q)
    cls.set_mcpaths = _set_mcpaths
    return cls

@up_ko_barrier
@down_ki_barrier
@mc_solvable
@pde_solvable
class snowball_option(OTC_option):
    def __init__(self,duration,coupon,strike=100,multiplier=100,CDS_swap=0.0,r=0.03,\
                 ki_flag=False,ko_count=0,days_per_year=365,rp=0.0,q=0.0):
        super(snowball_option,self).__init__(duration,coupon,strike,multiplier,CDS_swap,r,rp,q)
        self._ki_flag = ki_flag
        self._ko_count = ko_count
        self._days_per_year = days_per_year
    def set_coupon(self,coupon):
        self._coupon = coupon
    def set_ki_flag(self,flag):
        self._ki_flag = flag
    def set_up_ko_barrier(self,barriers,obs_dts,obs_type):
        super(snowball_option,self).set_up_ko_barrier(barriers,obs_dts,obs_type)
        self._after_ki_snowball_option.set_up_ko_barrier(barriers,obs_dts,obs_type)
    def price_value_via_mc(self):
        if '_paths' not in self.__dict__.keys():
            self._set_mcpaths()
        kwargs = {'strike':self._strike,'multiplier':self._multiplier,'strike_in_flag':self._ki_flag,\
                  'strike_out_count':self._ko_count,'r':self._r,'CDS_swap':self._CDS_swap,'rp':self._rp,'days_per_year':self._days_per_year}
        self._value_mc = snowball_value(self._paths,self._up_ko_barriers,self._down_ki_barriers,self._coupon,\
                                     self._up_ko_obs_dts,**kwargs)
    def price_greeks_via_mc(self,point=0.01):
        if '_paths' not in self.__dict__.keys():
            self._set_mcpaths()
        kwargs = {'strike':self._strike,'multiplier':self._multiplier,'strike_in_flag':self._ki_flag,\
                  'strike_out_count':self._ko_count,'r':self._r,'CDS_swap':self._CDS_swap,'rp':self._rp,'days_per_year':self._days_per_year}
        self._delta_mc = snowball_delta(self._paths,self._up_ko_barriers,self._down_ki_barriers,self._coupon,self._up_ko_obs_dts,\
                                     point=point,**kwargs)
    def set_bound_cond(self,func,bound_type):
        if bound_type=='up':
            self._up_bound_cond = func
        elif bound_type=='down':
            self._down_bound_cond = func
    def set_final_cond(self,func):
        self._final_cond = func
    def set_inside_cond(self,rows,cols,func,replace=False):
        if ('_inside_conds' not in self.__dict__.keys()) or replace:
            self._inside_conds = []
        self._inside_conds.append([rows,cols,func])
    def price_value_via_pde(self,down_bound_cond,up_bound_cond,inside_obs_cond,final_ki_cond,final_noki_cond):
        if '_after_ki_snowball_option' in list(self.__dict__.keys()):
            del self._after_ki_snowball_option
        
        self.set_bound_cond(down_bound_cond,'down')
        self.set_bound_cond(up_bound_cond,'up')
        self.set_final_cond(final_ki_cond)
        
        obs_dts_count = self._up_ko_obs_dts
        ub_dts = self._up_ko_barriers
        self._inside_conds = []
        self.set_mesh(S_max=np.log(self.S_max),S_min=np.log(self.S_min),M=self.M,set_inside=False)
        for i in range(len(obs_dts_count)-1):
            cols = np.array([obs_dts_count[i]])
            rows = np.argwhere(np.exp(self._xs)>=ub_dts[i]).flatten()
            self.set_inside_cond(rows,cols,inside_obs_cond,replace=False)
        self.set_mesh(S_max=np.log(self.S_max),S_min=np.log(self.S_min),M=self.M,set_inside=True)
        
        self._after_ki_snowball_option = deepcopy(self)
        self._after_ki_snowball_option.set_ki_flag(True)
        
        if not self._ki_flag:
            self._after_ki_snowball_option.solve_pde()
            if type(self._down_ki_barriers) == list:
                raise Exception('down_ki_barrier for snowball options should be of type float or int')
            
            # 更新
            self.set_final_cond(final_noki_cond)
            
            def myFunc(option,rows,cols):
                return option._after_ki_snowball_option._mesh.iloc[rows,cols]
            rows = np.argwhere(np.exp(self._xs)<=self._down_ki_barriers).flatten()[:-1]
            cols = [i for i in range(self._duration+1)]
            self._inside_conds.append([rows,cols,myFunc])
        
        self.set_mesh(S_max=self._mesh.index.max(),S_min=self._mesh.index.min(),M=self._M,set_inside=True)
        self.solve_pde()


#%%
def std_norm(n):
    '''
    利用Box-Muller转换生成标准正态分布的随机变量
    Input:
        - n(int): 需要生成的随机数个数
    Output:
        - list
    '''
    
    norm = (sqrt(-2*log(uniform(0,1)))*cos(2*pi*uniform(0,1)) for i in range(n))
    return list(norm)

def mc_paths(S0,sigma,n,r=0.03,M=50000,days_per_year=365,q=.0):
    '''
    生成模拟路径，假设价格服从几何布朗运动
    n 为 duration 转换为天数
    '''
    dt = 1./days_per_year
    paths = np.zeros((n+1,2*M))
    paths[0] = S0
    for i in range(1,n+1):
        rv = std_norm(M)
        paths[i,:M] = paths[i-1,:M]*np.exp((r-q-0.5*sigma**2)*dt+sigma*sqrt(dt)*np.array(rv))    
        paths[i,M:] = paths[i-1,M:]*np.exp((r-q-0.5*sigma**2)*dt-sigma*sqrt(dt)*np.array(rv))   
    return paths

def snowball_value(paths,upper,lower,coupon,obs_dts,strike=None,multiplier=100,\
                strike_in_flag=False,strike_out_count=0,days_per_year=365,\
                    r=0.03,rp=0.0,CDS_swap=0.0):
    paths_ = deepcopy(paths)
    if strike==None:
        strike = paths_[0][0]
    
    # 判断是否有敲入
    strike_in = strike_in_flag|np.any(paths_<lower,axis=0)
    
    # 判断是否有敲出，若有敲出则记录敲出日
    strike_out = np.zeros(shape=(1,paths_.shape[1]))
    duration = np.zeros(shape=(1,paths_.shape[1]))
    for i in range(len(obs_dts)):
        dt = obs_dts[i]
        prices = paths_[dt,(strike_out==0)[0]]
        idx = (strike_out==0)[0]
        print(idx)
        if type(upper)==float:
            strike_out[:,idx] = dt*(prices>=upper)
            duration[:,idx] = dt*(prices>=upper)
        elif type(upper)==list:
            strike_out[:,idx] = dt*(prices>=upper[i])
            duration[:,idx] = dt*(prices>=upper[i])
    duration[:,(strike_out==0)[0]] = paths.shape[0]
    
    interest = (strike_out_count*(strike_out!=0)+strike_out)/days_per_year*coupon
    premium = (strike_out_count+duration*(strike_out==0)+strike_out)/days_per_year*rp
    
    pnl = ((strike_in-strike_out)>0)*(paths_[-1,:]/strike-1)*(paths_[-1,:]<strike)+\
        interest+(strike_out+strike_in==0)*(coupon*(strike_out_count+duration)/days_per_year)+\
            premium+1
            
    return np.average(pnl/np.power(1+r+CDS_swap,duration/days_per_year))*multiplier

def snowball_delta(paths,upper,lower,coupon,obs_dts,point=0.01,**kwargs):
    # 利用MC得到的路径求delta
    '''
    calculate option delta
    Input:
        - point: pct change of underlying price
    Output:
        - delta: estimated delta
    '''
    multiplier = kwargs.get('multiplier',100)
    paths_plus = paths*(1+point)
    paths_sub = paths*(1-point)
    value_plus = snowball_value(paths_plus,upper,lower,coupon,obs_dts,**kwargs)
    value_sub = snowball_value(paths_sub,upper,lower,coupon,obs_dts,**kwargs)
    return (value_plus-value_sub)/(paths[0,0]*point*2)
