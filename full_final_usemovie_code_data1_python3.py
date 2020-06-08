# Final version made on 9th Oct : Variation 1 chop up method in death case
# Latest RJMCMC Code implementing Gibbs type of proposal sampler i.e. taking conditional distribution of user given movie feature or movie given user features instead of independent normal proposal sampler.
#Loss changed and collected properly

import pandas as pd
import numpy as np
import random as ran
import matplotlib.pyplot as plt
import time
import math
import scipy
from scipy.stats import norm

user_data=pd.read_csv('ml-100k/u.user', sep='|', header=None, encoding='latin-1')

movies_data=pd.read_csv('ml-100k/u.item', sep='|', header=None, encoding='latin-1')

ratings=pd.read_csv('ml-100k/u.data', sep='\t', header=None, encoding='latin-1')

print("The data has been taken")
#train=pd.read_csv('m-100k/ml-100k/u1.base',sep='\t',header=None,encoding='latin-1')
#test=pd.read_csv('m-100k/ml-100k/u1.test',sep='\t',header=None,encoding='latin-1')

nmovies=movies_data.shape[0]
nusers=user_data.shape[0]

ratings_matrix=np.zeros((nusers,nmovies))

train_mat=np.zeros((nusers,nmovies))

for i in range(1, 80000):
 train_mat[ratings[0][i]-1][ratings[1][i]-1]=ratings[2][i]
	
test_mat=np.zeros((nusers,nmovies))

for i in range(80000, 100000):
 test_mat[ratings[0][i]-1][ratings[1][i]-1]=ratings[2][i]
 
print("The train test split is 80 20")

init_k=25

print("The initial no of latent features is 25")
#normalization of ratings

train_rate01=(train_mat!=0)
train_rate02=(train_mat==0)

test_rate01=(test_mat!=0)
test_rate02=(test_mat==0)
   
#end of train_normalization

#train_mat=normalize_test_ratings(train_mat,train_rate01)

#test_mat=normalize_test_ratings(test_mat,test_rate01)

row_mean_train=np.zeros(shape=(nusers, 1))
train_rat_norm=np.zeros(shape=train_mat.shape)
#for i in range(0,nmovies):
 #idx = (train_rate01[i] == 1)[0]
 
# if(idx!=0):
  #rat_mean[i]=np.mean(train_mat[i,idx])
 # train_rat_norm[i,idx]=train_mat[i,idx]-rat_mean[i]
 #return rat_norm

#train_mat=train_rat_norm

#row_mean_train=0
for i in range(0, nusers):
 row_mean_train[i]=np.mean(train_mat[i])
 #print("\n\n\n Row Means", row_mean_train[i]) 
 for j in range(0, nmovies):
  train_rat_norm[i][j] =train_mat[i][j]-row_mean_train[i]
#   train_rat_norm[i][j] =train_mat[i][j]
# row_mean=0 
  
train_mat=train_rat_norm
   
#def normailze_test_ratings(mat,rate01):
row_mean_test=np.zeros(shape=(nusers, 1))
test_rat_norm=np.zeros(shape=test_mat.shape)
#for i in range(0,nmovies):
 #idx=(test_rate01[i]==1)[0]
 #if(len(idx)!=0):
 #rat_mean[i]=np.mean(test_mat[i,idx])  
 #test_rat_norm[i,idx]=test_mat[i,idx]-rat_mean[i]
#return rat_norm
#adam parameters

#test_mat=test_rat_norm

#row_mean_train=0
for i in range(0, nusers):
 row_mean_test[i]=np.mean(train_mat[i])
 for j in range(0, nmovies):
  test_rat_norm[i][j]=test_mat[i][j]-row_mean_test[i]
#   test_rat_norm[i][j]=test_mat[i][j]
 row_mean=0
 
test_mat=test_rat_norm

m1=0
v1=0
m2=0
v2=0
beta1=0.9
beta2=0.999
#alpha=0.001
#alpha=0.00001
alpha=0.0001
epsilon=0.000000001
t1=1
t2=1


def adam_hyp_update(theta,t,m,v,T,hyp_param):
 
 dim1=theta.shape[0]
 dim2=theta.shape[1]
 #for i in range(0,dim1):
  #for j in range(0,dim2):
   
 f=np.linalg.norm(theta,'fro')
 H=f*f/T
 H=H/(dim1*dim2)
 m=beta1*m+(1-beta1)*H
 v=beta2*m+(1-beta2)*H*H
 mhat=m/(1-beta1**t)
 vhat=v/(1-beta2**t)
 hyp_param=hyp_param-alpha*mhat/(math.sqrt(vhat)+epsilon)
 t=t+1
 print("In adam update")
 print("updated hyp parameter= ",hyp_param,"mean= ",m,"variance= ",v)
 return hyp_param,m,v,t
 
#def initialize_UV(k, n, p, T):
# M = 3.0
# S = 1.0
# mu = np.sqrt(M/k)
# sigma = np.sqrt(T/(np.sqrt(mu**4 + S/k) -mu**2))
# U = np.random.normal(mu,sigma,n*k).reshape(n,k)
# V = np.random.normal(mu,sigma,p*k).reshape(p,k)
# return U,V

def loss_nozro(M,U,V,l1,l2):
 loss = l1*np.linalg.norm(U,ord='fro')**2 + l2*np.linalg.norm(V,ord='fro')**2
 M_ = np.dot(U,V.T)
 nz = np.nonzero(M)
 loss += np.mean((M[nz] - M_[nz])**2)
 return loss
 
def unroll_params(X_and_theta,num_users,num_movies,k):
 first=X_and_theta[:int(num_movies*k)]
 X=first.reshape(int(k),int(num_movies)).transpose()
 last=X_and_theta[int(num_users*k)]
 theta=last.reshape(int(k),int(num_users)).transpose()
 #print("In adam update")
 #print()
 return X,theta
 

def loss_func(U, V, train_mat, num_users, num_movies, k, lamda1, lamda2, rate01):
 #U,V=unroll_params(X_and_theta,num_users,num_movies,k)
 #L=sum((V.dot(U.T)-train_mat)**2)/2
 MM_=U.dot(V.T)

 nz = np.nonzero(train_mat)
 L = np.mean((train_mat[nz] - MM_[nz])**2)
 #H_U=0
 #H_V=0
 #for i in range(0,nusers):
  #for j in range(0,k):
   #H_U=H_U+U[i][j]*U[i][j]
 #for i in range(0,nmovies):
  #for j in range(0,k):
   #H_V=H_V+V[i][j]*V[i][j]
 
 
 regu_1=lamda1*(np.linalg.norm(U,'fro')**2)

# regu_1=lamda1*(np.linalg.norm(U,'fro')**2)

 regu_2=lamda2*(np.linalg.norm(V,'fro')**2)

# regu_2=lamda2*(np.linalg.norm(V,'fro')**2)

 L=L+(regu_1+regu_2)
 return L
 
 
def loss_only(U,V,train_mat,num_users,num_movies,k,lamda1,lamda2,rate01):
 #U,V=unroll_params(X_and_theta,num_users,num_movies,k)
 #L=sum((U.dot(V.T)*rate01-train_mat)**2)/2
 MM_=U.dot(V.T)

 nz = np.nonzero(train_mat)
 L = np.mean((train_mat[nz] - MM_[nz])**2)
 #regu_1=lamda1*(np.linalg(U,'fro')**2)
 #regu_2=lamda2*(np.linalg(V,'fro')**2)
 #L=L-(regu_1+regu_2)
 return L

def loggu(data_u, data_v, Temp, regu_param1, regu_param2):

 n1 = len(data_u)
 n2 = len(data_v)
 user_stand_div=math.sqrt(abs(Temp/regu_param1))
 movie_stand_div=math.sqrt(abs(Temp/regu_param2)) 
 llogguser = -(n1/2)*math.log(2*math.pi) - n1*math.log(user_stand_div) - 0.5*sum((data_u/user_stand_div)**2)
 lloggmovie = -(n2/2)*math.log(2*math.pi) - n2*math.log(movie_stand_div) - 0.5*sum((data_v/movie_stand_div)**2) 

 return llogguser, lloggmovie, llogguser + lloggmovie 
 
#regu_param1=25
#regu_param2=25
#Temp=306.25

regu_param1=440
regu_param2=440
Temp=306.25

#regu_param1=9
#regu_param2=9
#Temp=18

adam_terminate=0

#U_V=np.zeros(((nusers+nmovies)*init_k))
#for i in range(0,(nusers+nmovies)*init_k):
#U_V=ran.random()

#U,V=unroll_params(U_V,nusers,nmovies,init_k)
#unroll_params(X_and_theta,num_users,num_movies,k)

#U=np.zeros((nusers,init_k))
#V=np.zeros((nmovies,init_k))

stand_div_users=math.sqrt(Temp/regu_param1)
stand_div_movies=math.sqrt(Temp/regu_param2)

#U=norm.rvs(size=(nusers,init_k),loc=np.sqrt(3/init_k),scale=stand_div_users)
#V=norm.rvs(size=(nmovies,init_k),loc=np.sqrt(3/init_k),scale=stand_div_movies)
U=norm.rvs(size=(nusers,init_k),loc=0,scale=stand_div_users)
V=norm.rvs(size=(nmovies,init_k),loc=0,scale=stand_div_movies)
#U, V = initialize_UV(init_k, nusers, nmovies)

min_k=init_k
c=0

sim_ana_param=0.99954306099
#sim_ana_param=0.99915306099
#sim_ana_param=0.999915306099
regu_prev1=0
regu_prev2=0
t1=1
t2=1

mm = 1000

k=init_k
error=np.zeros((mm))
train_error=np.zeros((mm))
k_val=np.zeros((mm))

start_time=time.time()

user_points= []
movie_points= []
hyp_param1=np.zeros((mm))
hyp_param2=np.zeros((mm))

test_wrt_k=np.repeat(9999, mm)
train_wrt_k=np.repeat(9999, mm)
k_freq=np.zeros((mm))

while c<mm:
 gu=1

 #mink=25

 d_b=ran.uniform(0,3)
 #if(c>500):
 # mink,k2=np.where(train_wrt_k==np.min(train_wrt_k[np.nonzero(train_wrt_k)]))
 #if(k<k2):
 #   d_b=ran.uniform(1,2)
 #elif(k>k2):
 #   d_b=ran.uniform(0,1)
 #else:
 #   d_b=ran.uniform(2,3)
 if(k<2):
  #error[c]=loss_only(U,V,train_mat,nusers,nmovies,k,regu_param1,regu_param2,train_rate01)
  #k_val[c]=k
  #user_points[c]=U
  #movie_points[c]=V
  #print(error[c])
  #U_point=np.zeros((nusers,50))
  #V_point=np.zeros((nmovies,50))
  
  #a=np.pad(U,(0,(50-k)),'constant',constant_values=0)
  #b=np.pad(V,(0,(50-k)),'constant',constant_values=0)
  #user_points[c]=a[0:nusers,:]
  #movie_points[c]=b[0:nmovies,:]
  #c=c+1
  #Temp=Temp*sim_ana_param
  d_b=ran.uniform(1, 2)
  #continue
  
 #d_b=ran.random()*3
 #death_birth_stay probability
 if(k>49):
  d_b=ran.uniform(0, 1)

 r=ran.uniform(0, 1)
 if (d_b<=2 and d_b>1):
  #birth move from init_k to init_k+1
  #new_user_data=np.zeros((nusers,1))
  #user_stand_div=math.sqrt(abs(Temp/regu_param1))
  #new_user_data=norm.rvs(size=nusers,loc=0,scale=user_stand_div)  
 
  total=0
  #for i in range(0,nusers):
   #total=total+abs(new_user_data[i])
  # new_user_data=new_user_data/total
  # total=0
  #new_user_data=np.reshape(new_user_data,(nusers,1))
  new_user_feature=np.zeros((nusers, k+1))
  #for i in range(0,nusers):
   #for j in range(0,k):
   #new_user_feature[j][i]=U[j][i]
  # for i in range(0,nusers):
  # new_user_feature[k][i]=new_user_data[k]
  
  #new_user_feature=np.hstack((U,new_user_data))
  #print(new_user_feature.shape[0],new_user_feature.shape[1])
  #print(new_user_feature)
  
  #movie_stand_div=math.sqrt(abs(Temp/regu_param2))
  
  b = np.zeros(nusers);
  new_user_data = np.zeros(nusers);
  a = 0
 
# print("\n\n\n",  sum((V[0:nmovies, k-1])^2))

  for i in range(0, nusers):
    b[i] = (-(np.matrix((V.T)[k -1, 0:nmovies])).dot((np.matrix(np.matrix(U[i, 0:k]).dot(V.T) - U[i, k -1]*V.T[k -1, 0:nmovies])).T) + np.matrix(train_mat[i, :]).dot(V[0:nmovies, k -1]))*(1.0/(nusers*nmovies))*(1/Temp)
    a = (1.0/(nusers*nmovies))*(sum((V[0:nmovies, k-1])**2))*(1.0/Temp) + ((regu_param1))*(1.0/Temp)
    #print("\n\n\n\n", a)
    new_user_data[i] =norm.rvs(loc=((b[i])/a), scale= math.sqrt(1.0/a), size=1)

  #print("\n\n\n\n", U.shape,'  ', new_user_data.shape)
  new_user_data=np.reshape(new_user_data, (nusers, 1))
  new_user_feature=np.hstack((U, new_user_data))

  #new_movie_data=norm.rvs(size=nmovies,loc=0,scale=movie_stand_div)
  #for i in range(0,nmovies):
  # total=total+abs(new_movie_data[i])
  #new_movie_data=new_movie_data/total
  #total=0
  #new_movie_feature=np.zeros((nmovies,k+1))
  #for i in range(0,nmovies):
   #for j in range(0,k):
   # new_movie_feature[j][i]=V[j][i]
  #for i in range(0,nmovies):
  # new_movie_feature[k][i]=new_movie_data[i]
  #applying the transformation such that |J|=1

  #new_movie_data=np.reshape(new_movie_data,(nmovies,1))
  
  #print("\n\n\n", np.matrix(V).sum(), (train_mat[:][1].shape[0]), (U[0:nusers, k-1]).shape, nusers)

  cc = np.zeros(nmovies);
  new_movie_data = np.zeros(nmovies);
  d = 0; 

#  print("\n\n\n",  np.matrix(U.T[1, 0:nusers]).shape, U.shape, (np.matrix(V.T[0:k, 1])).shape)

# Change of Proposal density 

  for j in range(0, nmovies):   
   cc[j] = (-np.matrix(U.T[k - 1, 0:nusers]).dot(np.matrix(np.matrix(U).dot(V.T[0:k, j]) - U[0:nusers, k -1]*V.T[k -1, j]).T) + np.matrix((train_mat[:, j]).T).dot(U[0:nusers, k -1]))*(1.0/(nusers*nmovies))*(1.0/Temp)
   d = (1.0/(nusers*nmovies))*(sum((U[0:nusers, k -1])**2))*(1.0/Temp) + (regu_param2)*(1.0/Temp)
   new_movie_data[j] =norm.rvs(loc=((cc[j])/d), scale= math.sqrt(1.0/d), size=1)

  new_movie_data=np.reshape(new_movie_data,(nmovies,1))
  new_movie_feature=np.hstack((V, new_movie_data))

  for i in range(0,nusers):
   temp1=(new_user_feature[i][k - 1]+new_user_feature[i][k])/math.sqrt(2)
   temp2=(new_user_feature[i][k - 1]-new_user_feature[i][k])/math.sqrt(2)
   new_user_feature[i][k - 1]=temp1
   new_user_feature[i][k]=temp2
   
  for i in range(0,nmovies):
   temp1=(new_movie_feature[i][k - 1]+new_movie_feature[i][k])/math.sqrt(2)
   temp2=(new_movie_feature[i][k - 1]-new_movie_feature[i][k])/math.sqrt(2)
   new_movie_feature[i][k - 1]=temp1
   new_movie_feature[i][k]=temp2
  gu=1
  # for i in range(0,nusers):
  # gu=gu*norm(0,user_stand_div).pdf(new_user_data[i])
  # for i in range(0,nmovies):
  # gu=gu*norm(0,movie_stand_div).pdf(new_movie_data[i])
   
  #lossdata_u, lossdata_v, lossdata_uplusv = loggu(norm.rvs(size=nusers,loc=0,scale=user_stand_div), norm.rvs(size=nmovies,loc=0,scale=movie_stand_div), Temp, regu_param1, regu_param2);

  #lo_accept_prob=0 
  if(gu!=0):
   new_loss=loss_func(new_user_feature[:,0:k],new_movie_feature[:,0:k],train_mat,nusers,nmovies,k,regu_param1,regu_param2,train_rate01);
   #loss_func(U,V,train_mat,num_users,num_movies,k,lamda1,lamda2,rate01):
   old_loss=loss_func(U,V,train_mat,nusers,nmovies,k,regu_param1,regu_param2,train_rate01);
   #print(old_loss)
   lo_accept_prob=((old_loss-new_loss)/Temp)
   #lo_accept_prob=accept_prob/gu
   #accept_prob=accept_prob*(death_prob[k+1]/birth_prob[k])
   
  # if (lo_accept_prob>0):
  #  lo_accept_prob=0
  
  if new_loss < old_loss or math.log(r) < lo_accept_prob:
   U=new_user_feature;
   V=new_movie_feature;
   k=k+1
   elapsed_time=time.time()-start_time
   print('accepting birth   ',U.shape,'    ',V.shape,'   k value= ',k,' time elapsed= ',elapsed_time)
   #print(U.shape)
   #print(V.shape)
   #print(U)
   #print(V)   
   
   regu_prev1=regu_param1
   regu_prev2=regu_param2
   #adam_hyp_update(theta,t,m,v,T,hyp_param)
   if(adam_terminate!=1):
    regu_param1,m1,v1,t1=adam_hyp_update(U,t1,m1,v1,Temp,regu_param1)
    regu_param2,m2,v2,t2=adam_hyp_update(V,t2,m2,v2,Temp,regu_param2)
   if(abs(regu_param1-regu_prev1)<0.00000001 or abs(regu_param2-regu_prev2)<0.00000001):
    adam_terminate=1
   else:
     regu_prev1=regu_param1
     regu_prev2=regu_param2
  elif(math.log(r) >= lo_accept_prob):
   elapsed_time=time.time()-start_time
   print('keeping same   ', U.shape,'     ', V.shape,'   k value= ', k,' time elapsed= ', elapsed_time)
  error[c]=loss_func(U, V, test_mat, nusers, nmovies, k, regu_param1, regu_param2, test_rate01)
  train_error[c]=loss_func(U, V, train_mat, nusers, nmovies, k, regu_param1, regu_param2, train_rate01)
  hyp_param1[c]=regu_param1
  hyp_param2[c]=regu_param2
  if(error[c]<test_wrt_k[c]):
    test_wrt_k[c]=error[c]
    train_wrt_k[c]=train_error[c]
    #test_wrt_k[k-1]=(test_wrt_k[k-1]*k_freq[k-1]+error[c])/(k_freq[k-1]+1)
    #train_wrt_k[k-1]=(train_wrt_k[k-1]*k_freq[k-1]+train_error[c])/(k_freq[k-1]+1)
  k_freq[c]=k_freq[c]+1

  print(error[c])
  k_val[c]=k
  #U_point=np.zeros((nusers,50))
  #V_point=np.zeros((nmovies,50))

  user_points.append(np.array(U))
  movie_points.append(np.array(V))
    
  #a=np.pad(U,(0,(50-k)),'constant',constant_values=0)
  #b=np.pad(V,(0,(50-k)),'constant',constant_values=0)
  #user_points[k-1]=a[0:nusers,:]
  #movie_points[k-1]=b[0:nmovies,:]
  c=c+1
  Temp=Temp*sim_ana_param

 #death move
 elif(d_b<=1):
  old_U=np.zeros((nusers, k))
  old_V=np.zeros((nmovies, k))
  old_U=U
  old_V=V
  for i in range(0,nusers):
   temp1=(U[i][k-1]+U[i][k-2])/math.sqrt(2)
   temp2=(U[i][k-1]-U[i][k-2])/math.sqrt(2)
   U[i][k-2]=temp1
   U[i][k-1]=temp2
  for i in range(0,nmovies):
   temp1=(V[i][k-1]+V[i][k-2])/math.sqrt(2)
   temp2=(V[i][k-1]-V[i][k-2])/math.sqrt(2)
   V[i][k-2]=temp1
   V[i][k-1]=temp2
  gu=1

  #user_stand_div=math.sqrt(abs(Temp/regu_param1))
  #movie_stand_div=math.sqrt(abs(Temp/regu_param2))
  #for i in range(0,nusers):
   #for j in range(0,k):
    #gu=gu*norm(0,user_stand_div).pdf(U[i][j])
  
  #for i in range(0,nmovies):
  # for j in range(0,k):
   # gu=gu*norm(0,movie_stand_div).pdf(V[i][j])
  
  new_U=np.zeros((nusers,k-1))
  new_V=np.zeros((nmovies,k-1))
  
  #for i in range(0,nusers):
   #for j in range(0,k-1):
   # new_U[i][j]=U[i][j]
  #for i in range(0,nmovies):
   #for j in range(0,k-1):
   # new_V[i][j]=V[i][j]
  
  new_U=U[:,0:k-1]
  new_V=V[:,0:k-1]
    
  U=old_U
  V=old_V
  old_loss=loss_func(U[:,0:k-1], V[:,0:k-1], train_mat,nusers,nmovies,k - 1,regu_param1,regu_param2, train_rate01)
  new_loss=loss_func(new_U,new_V,train_mat,nusers,nmovies,k - 1,regu_param1, regu_param2, train_rate01)
  #loss_func(U,V,train_mat,num_users,num_movies,k,lamda1,lamda2,rate01):
  lo_accept_prob=0
  #if(gu!=0):
  lo_accept_prob=((old_loss-new_loss)/Temp)
   #accept_prob=accept_prob*gu
  # accept_prob=accept_prob*(birth_prob[k-1]/death_prob[k])
  #if(lo_accept_prob>0):
  # lo_accept_prob=0
   
  if new_loss < old_loss or math.log(r) < lo_accept_prob:
   U=new_U
   V=new_V
   k=k-1
   time_elapsed=time.time()-start_time
   print("accepting death   ",U.shape,"    ",V.shape,"   k value= ",k," time elapsed= ",time_elapsed)
   #print(U.shape)
   #print(V.shape)
   
   regu_prev1=regu_param1
   regu_prev2=regu_param2
   
   #adam update
   
   if(adam_terminate!=1):
    regu_param1,m1,v1,t1=adam_hyp_update(U,t1,m1,v1,Temp,regu_param1)
   if(adam_terminate!=1):
    regu_param2,m2,v2,t2=adam_hyp_update(V,t2,m2,v2,Temp,regu_param2)
   
   if(abs(regu_prev1-regu_param1)<0.00000001 or abs(regu_prev2-regu_param2)<0.00000001):
    adam_terminate=1
   else:
     regu_prev1=regu_param1
     regu_prev2=regu_param2
  elif(math.log(r) >= lo_accept_prob):
   elapsed_time=time.time()-start_time
   print('keeping same   ',U.shape,'    ',V.shape,'   k value= ',k,' time elapsed= ',time_elapsed)
  error[c]=loss_func(U,V,test_mat,nusers,nmovies,k,regu_param1,regu_param2,test_rate01)
  train_error[c]=loss_func(U,V,train_mat,nusers,nmovies,k,regu_param1,regu_param2,train_rate01)
  hyp_param1[c]=regu_param1
  hyp_param2[c]=regu_param2
  k_val[c]=k

  if(error[c]<test_wrt_k[c]):
   test_wrt_k[c]=error[c]
   train_wrt_k[c]=error[c]
   #test_wrt_k[k-1]=(test_wrt_k[k-1]*k_freq[k-1]+error[c])/(k_freq[k-1]+1)
   #train_wrt_k[k-1]=(train_wrt_k[k-1]*k_freq[k-1]+train_error[c])/(k_freq[k-1]+1)
  k_freq[c - 1]=k_freq[c - 1]+1

  user_points.append(np.array(U))
  movie_points.append(np.array(V))
  
  print(error[c])
  #U_point=np.zeros((nusers,50))
  #V_point=np.zeros((nmovies,50))
  
  #a=np.pad(U,(0,(50-k)),'constant',constant_values=0)
  #b=np.pad(V,(0,(50-k)),'constant',constant_values=0)
  #user_points[k-1]=a[0:nusers,:]
  #movie_points[k-1]=b[0:nmovies,:]
  c=c+1
  Temp=Temp*sim_ana_param
  	
 else:
  #print('no birth no death')
  user_stand_div=math.sqrt(Temp/regu_param1)
  movie_stand_div=math.sqrt(Temp/regu_param2)
  #stay at the same dimension 
  #using MH algorithm to calculate the acceptance of the new point or keep the old point
  #new_U=norm.rvs(size=(nusers,k),loc=np.sqrt(3.0/k),scale=stand_div_users)
  #new_V=norm.rvs(size=(nmovies,k),loc=np.sqrt(3.0/k),scale=stand_div_movies)
  new_U=norm.rvs(size=(nusers,k),loc=0,scale=user_stand_div)
  new_V=norm.rvs(size=(nmovies,k),loc=0,scale=movie_stand_div)
  #print(new_U.shape)
  #print(new_V.shape)
  #print(new_U.shape)
  #print(new_V.shape)
  new_loss=loss_func(new_U,new_V,train_mat,nusers,nmovies,k,regu_param1,regu_param2,train_rate01)
  old_loss=loss_func(U,V,train_mat,nusers,nmovies,k,regu_param1,regu_param2,train_rate01)
  #print(old_loss.shape)
  #print(new_loss.shape)
  loss_diff=old_loss-new_loss
  loss_diff=loss_diff/Temp
  lo_accept_prob=(loss_diff/Temp)
  #if(lo_accept_prob > 0):
  # lo_accept_prob=0
  
  if(new_loss < old_loss or math.log(r) < lo_accept_prob):   
   U=new_U
   V=new_V
   time_elapsed=time.time()-start_time
   print('accepting new point of same dimension  ', U.shape,'   ', V.shape,'   k value= ', k,' time elapsed= ')
   #print(U.shape)
   #print(V.shape)
   #adam's hyperparameter update
   regu_prev1=regu_param1
   regu_prev2=regu_param2
   if(adam_terminate!=1):
    regu_param1,m1,v1,t1=adam_hyp_update(U,t1,m1,v1,Temp,regu_param1)
    regu_param2,m2,v2,t2=adam_hyp_update(V,t2,m2,v2,Temp,regu_param2)
   if(abs(regu_prev1-regu_param1)<0.00000001 or abs(regu_prev2-regu_param2)<0.00000001):
    adam_terminate=1
   else:
     regu_prev1=regu_param1
     regu_prev2=regu_param2 
  #elif(math.log(r) >= lo_accept_prob):
   time_elapsed=time.time()-start_time
   print('keeping same point   ',U.shape,'   ',V.shape,'   k value= ',k,' time elapsed= ',time_elapsed)
  error[c]=loss_func(U,V,test_mat,nusers,nmovies,k,regu_param1,regu_param2,test_rate01)
  train_error[c]=loss_func(U,V,train_mat,nusers,nmovies,k,regu_param1,regu_param2,train_rate01)
  hyp_param1[c]=regu_param1
  hyp_param2[c]=regu_param2
  k_val[c]=k

  if(error[c]<test_wrt_k[c]):
   test_wrt_k[c]=error[c]
   train_wrt_k[c]=train_error[c]
  k_freq[c]=k_freq[c]+1

  user_points.append(np.array(U))
  movie_points.append(np.array(V))

  #print(error[c])
  #U_point=np.zeros((nusers,50))
  #V_point=np.zeros((nmovies,50))
  
  #a=np.pad(U,(0,(50-k)),'constant',constant_values=0)
  #b=np.pad(V,(0,(50-k)),'constant',constant_values=0)
  #user_points[c-1]=U[0:nusers,:]
  #movie_points[c-1]=V[0:nmovies,:]
  c=c+1
  Temp=Temp*sim_ana_param
        
#adam_hyp_update(theta,t,m,v,T,hyp_param):
#loss_func(U,V,train_mat,num_users,num_movies,k,lamda1,lamda2,rate01):
  
#end_of_simulated annealing

#print(k)
#print(c)
   
min_error=999999999
k_opt=0
index=0

iteration=range(1, mm+1)
k_v=range(1, 190)
#iteration=np.zeros((mm))
#for i in range(0,mm):
#	iteration[i]=i+1
#k_v=np.zeros((50))
#for i in range(0,50):
#	k_v[i]=i+1

#implementing moving average

#mov_avg_error=np.zeros((190))
index=0
mve_least=999999
avg_iter_val=0
for i in range(0,mm):
  if(mve_least>error[i]):
    mve_least=error[i]
    #index=i
    k_opt=k_val[i]
    index = i + 1

print('Final Temparature is: ', Temp)
print('The optimum number of latent features based on minimizing test set errors is: ', k_opt)


for i in range(0, mm):
  if(test_wrt_k[i]==9999):
    test_wrt_k[i]=0
    train_wrt_k[i]=0

# Taking a window of length 100 iterations and a stride of 10:

#mov_avg_error=np.zeros((190))

num1 = 200

test_wrt = np.zeros(int((mm - num1)/10))
k_opt_mov = np.zeros(int((mm - num1)/10))

index_lower=0
index_upper=0
minimum=9999999
for i in range(0,int((mm - num1)/10)):
  summation=(sum(error[10*i:10*i+num1])/num1)
  if(minimum>summation):
    minimum=summation
    test_wrt[i] = np.average(error[10*i:10*i+num1])
    index_lower=10*i
    index_upper=10*i+num1
    k_opt_mov[i]=np.average(k_val[index_lower:index_upper]) 

index_lower=int(index_lower)
index_upper=int(index_upper)
k_opt_win=np.average(k_val[index_lower:index_upper])

k_opt_win=int(k_opt_win) + 1

print('The optimum number of latent features based on minimum moving average test set error is: ', k_opt_win)

#k_opt=int(k_opt)
#for i in range(0, mm):
# if(min_error>test_wrt_k[i] and test_wrt_k[i]>=0):
#  #k_opt=k_val[i]
#  min_error=test_wrt_k[i]
#  index=i+1
#print('The optimum k value ',index)
#k_opt_alt=index

#print('The Index when optimum number of latent features occurs : ', k_opt_alt)

collect_index = []
collect_index1 = []

for i in range(0, mm):
 #print(user_points[i].shape[1], k_opt_win)
 if(user_points[i].shape[1] == k_opt_win):
   collect_index.append(i)
 if(movie_points[i].shape[1] == k_opt_win):
   collect_index1.append(i)

collect_index = np.array(collect_index, dtype='int32')
collect_index1 = np.array(collect_index1, dtype='int32')

#print(user_points[collect_index[1]])
print(collect_index[1])

UU = VV = 0

for i in range(0, len(collect_index)):
 UU += user_points[int(collect_index[i])]
for i in range(0, len(collect_index1)):
 VV += movie_points[int(collect_index1[i])]

U = UU/len(collect_index)
V = VV/len(collect_index1)

#final_test_error=loss_only(U,V,test_mat,nusers,nmovies,k_opt,regu_param1,regu_param2,test_rate01)
final_test_error= loss_nozro(test_mat,U,V,0,0)
#final_train_error=loss_only(U,V,train_mat,nusers,nmovies,k_opt,regu_param1,regu_param2,train_rate01)
#final_test_val_error=loss_func(U,V,test_mat,nusers,nmovies,k_opt,regu_param1,regu_param2,test_rate01) 
#loss_only(U,V,train_mat,num_users,num_movies,k,lamda1,lamda2,rate01):
print('The final test error thus obtained is ', math.sqrt(final_test_error))
#print('The final train error thus obtained for the identical k value ', final_train_error)
#print('The final train error thus obtained for the identical k value ', math.sqrt(final_test_val_error))
#print('The final test error thus obtained for the identical k value ', min(error))

#print('lambda1 finally converges to ', hyp_param1[index])
#print('lambda2 finally converges to ', hyp_param2[index])

#plt.plot(k_val,error)
#plt.plot(k_val,train_error)
f1=plt.figure()
plt.plot(iteration, error)
#plt.plot(iteration,train_error)
plt.xlabel('iterations')
plt.ylabel('Error')
plt.legend()
plt.savefig('iteration_vs_error_trial.png')
plt.savefig('iteration_vs_error_trial.eps')
plt.title('Error vs current iteration')

f2=plt.figure()
plt.plot(np.sort(k_opt_mov), test_wrt[np.argsort(k_opt_mov)])
#plt.plot(k_v,train_wrt_k)
plt.xlabel('value of k')
plt.ylabel('mean error at that point')
plt.legend()
plt.savefig('mav_vs_k_trial.png')
plt.savefig('mav_vs_k_trial.eps')
plt.title('Moving average error vs number of latent features')

f3=plt.figure()
plt.plot(iteration,k_val)
plt.xlabel('current iteration')
plt.ylabel('Number of latent features')
plt.savefig('iteration_vs_k_val_trial.png')
plt.savefig('iteration_vs_k_val_trial.eps')
plt.show()

#print(np.mean(k_val[1400:1750]))
   
   


