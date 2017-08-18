#encoding:utf-8

'''
@author:zhanghonglei
@email:hongleizhang@bjtu.edu.cn
@date:2017-08-18
'''

#inner library
import time
import math
#third library
import numpy as np
import pandas as pd
import scipy.sparse as ss

class BiasSVD(object):
	"""docstring for BiasSVD"""
	def __init__(self, path,USER_NUM,ITEM_NUM,FACTOR):
		super(BiasSVD, self).__init__()
		self.path = path
		self.USER_NUM=USER_NUM
		self.ITEM_NUM=ITEM_NUM
		self.FACTOR=FACTOR
		self.init_model()
		

	def init_model(self):

		self.P=np.random.rand(self.USER_NUM, self.FACTOR) / (self.FACTOR ** 0.5)
		self.Q=np.random.rand(self.FACTOR, self.ITEM_NUM) / (self.FACTOR ** 0.5)
		self.bu=np.zeros((self.USER_NUM))
		self.bi=np.zeros((self.ITEM_NUM))
		self.m=0
		pass

	#从txt文件中读取数据，返回DataFrame,Matrix,Array格式
	def load_data(self,file_path, file_type='txt', sep='\t', header=None, return_format='DataFrame'):
		"""
		加载数据文件，返回指定类型 
		@@zhanghonglei

		Args:
			file_path: 数据文件存放路径
			file_type: 数据文件格式csv,dat,txt,json
			sep: 分隔字符串'\t',','
			header: 数据文件标题，0-表示从第1行开始，None-表示数据文件中无标题
			return_format: 返回的数据类型，返回DataFrame,Matrix,Array格式

		Returns:
			data: 返回指定格式的数据

		"""
		if file_type == 'csv':
			data=pd.read_csv(file_path, header=header)
		elif file_type =='dat':
			data=pd.read_csv(file_path, sep=sep, header=header)
		elif file_type == 'txt':
			data=pd.read_table(file_path, sep=sep, header=header) # encoding='iso-8859-1'
		elif file_type =='json': #读取后缀名为.json的文件,默认返回dataframe格式；若不加lines=True的话，会报Trailing data的错误
			data=pd.read_json(file_path,lines=True)
		else:
			print('暂不支持此文件类型')

		if return_format == 'Matrix':
			data=np.mat(data)
		if return_format == 'Array':
			data=np.array(data)
		return data

	#将DataFrame格式的数据转换为Rating Matrix 慢     #转换为矩阵的时候都减了1
	def frame_to_mat(self,data,row_num,column_num):
		sparse_mat=ss.dok_matrix((row_num,column_num))
		for index,row in data.iterrows():
			user, item, rating = data.ix[index, 0], data.ix[index, 1], data.ix[index, 2]
			sparse_mat[int(user-1), int(item-1)]=int(rating)
		return sparse_mat

	def train(self, data, epochs=500, theta=1e-4, alpha=0.02, beta=0.02 ):
		"""
		带偏置项的矩阵分解 biasSVD

		Args:
			data: 需要进行分解的矩阵R
			f: 隐含因子的个数
			e: 误差的预制，默认为0.001
			epochs: 最大迭代次数
			alpha: 学习率
			beta: 正则化系数
		Returns:
			返回P,Q矩阵

		"""
		print("开始运算")
		self.m=data.sum()/len(data.items()) #评分均值 
		print("平均分计算完毕{}",m)
		data_nonzero=data.nonzero()
		data_zip=list(zip(data_nonzero[0],data_nonzero[1])) #zip 在py2中返回list,在py3中返回迭代器
		print("开始迭代")
		eb=0
		es=[]
		for current_epoch in range(epochs):
			print("第{}次迭代",current_time)
			e=0
			# ss=time.clock()
			for item in range(len(data_zip)): #空读0.02 稀疏矩阵计算0.03603109688992845 数组计算0.0010
				x, y=data_zip[item]
				r=data[x,y]
				dot=np.dot(self.P[x,:],self.Q[:,y])
				#计算预测评分误差
				err=r-(m+self.bu[x]+self.bi[y]+dot)
				e+=pow(err,2)

				#更新偏置项
				self.bu[x]+=alpha*(err-beta*self.bu[x])
				self.bi[y]+=alpha*(err-beta*self.bi[y])
				#更新P,Q
				for i in range(f):  
					self.P[x,i]+=alpha*(err*self.Q[i,y]-beta*self.P[x,i])
					self.Q[i,y]+=alpha*(err*self.P[x,i]-beta*self.Q[i,y])
					#计算损失函数
					e+=(beta/2)*(pow(self.P[x,i],2)+pow(self.Q[i,y],2)) #需不需要计算正则项的误差
			print("损失值e={}",e)
			alpha*=0.9
			#设置阈值停止迭代
			if np.abs(e-eb)<t:
				print("到达阈值终止条件{}",e)
				break
			es.append(e) #为了画图
			eb=e #更新eb
		print("到达迭代终止条件{}",current_epoch)
