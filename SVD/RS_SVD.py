from scipy.sparse import coo_matrix,dok_matrix
import os
import math
import random
import numpy as np
from matplotlib import pyplot as plt
import datetime
import sys
sys.setrecursionlimit(100000000)

start = datetime.datetime.now()
row=[]
col=[]
score=[]
max_item=0

###################################################
####               数据预处理                   ####
####              在内存中存入：                #### 
####         user-item-score的numpy数组        ####
###################################################

with open("train.txt","r",encoding="utf8") as f:
    for line in f.readlines():
        if line=="\n" :
            break
        if "|" in line:
            user=int(line.split("|")[0])
            continue
        else:
            row.append(user)
            col.append(int(line.split()[0]))
            score.append(int(line.split()[1]))
            if int(line.split()[0])>max_item:
                max_item=int(line.split()[0])

print("phase1 end")

new_usernum=user+1
print("user num:",new_usernum)
new_itemnum=max_item+1
print("item num:",new_itemnum)

new_data=np.vstack((np.array(row),np.array(col)))
new_data=np.vstack((new_data,np.array(score))).T




###################################################
####               数据预处理                   ####
####            shuffle打乱数据                 #### 
####         划分为训练集：验证集 = 9：1         ####
###################################################


random.shuffle(new_data)
print("data shape",new_data.shape)

train_num=int(0.9*len(new_data))
print("training_set number:",train_num)
traindata=new_data[:train_num,:]
testdata=new_data[train_num:,:]

Average_score=np.average(score)
print("parse2 end, Average_score is",Average_score)





###################################################
####               训练准备                    ####
###################################################


'''
    一维数组以矩阵乘法相乘
    v1,v2是一维向量
    返回v1,v2逐元素乘积和
'''
def mult_1d(v1, v2):
	result = 0
	for i in range(len(v1)):
		result += v1[i] * v2[i]
	return result
	
'''
    预测用户对电影的评分
'''
def predict(aver, bu, bi, qi, px):
	res = aver + bu + bi + mult_1d(qi, px)
	if res < 0:
		res = 0
	if res > 100:
		res = 100
	return  res

'''
    验证集上计算RMSE
'''
def validate(testData, aver, bu, bi, qi, px):
	rmse = 0.0
	for i in range(len(testData)):
		user = testData[i][0]
		item = testData[i][1]
		score_hat = predict(aver, bu[user], bi[item], qi[item], px[user])
		score_gt = testData[i][2]
		rmse += (score_gt - score_hat) * (score_gt - score_hat)
	return math.sqrt(rmse / len(testData))


'''
    test.txt上模型推理，计算评分
'''
def test(aver,bu,bi,qi,pu):
	user=0
	item=0
	score=0
	with open("test.txt","r",encoding="utf8") as f1:
		with open("resul_svd.txt","w",encoding="utf8") as f2:
			for line in  f1.readlines():
				if line=="\n":
					break
				if "|" in line:
					f2.write(line)
					user=int(line.split("|")[0])
					continue
				else:
					item=int(line.split()[0])
					score=int(predict(aver, bu[user], bi[item], qi[item], pu[user]))
					f2.write(str(item)+" "+str(score)+"\n")
			

'''
    SVD训练、推理过程
    输入参数：
    factorNum: 特征维度
    Lr:   学习率
    reg:  正则参数     
'''
def SVD(testData, trainData, factorNum, Lr, reg):
    
    #参数初始化：

	aver = Average_score
	userNum = new_usernum
	itemNum = new_itemnum
	bi = np.zeros(itemNum)
	bu = np.zeros(userNum)
	qi = np.zeros((itemNum,factorNum))
	pu = np.zeros((userNum,factorNum))
	# qi = dok_matrix((itemNum,factorNum),dtype=np.float)
	# pu = dok_matrix((userNum,factorNum),dtype=np.float)
	last_Rmse = 10000.0
	Epochs = 35

    #开始训练
	print("Training begins:")
	for epoch in range(Epochs):
		print("Epoch",epoch," begins")
		time1=datetime.datetime.now()
        #训练进度显示
		for i in range(len(trainData)):
			if i%int(len(trainData)/5)==0:
				print(int(100*i/len(trainData)),"% of current epoch ends")
			user = trainData[i][0]
			item = trainData[i][1]
			score = trainData[i][2]	
            
            #计算评分	
			prediction = predict(aver, bu[user], bi[item], qi[item], pu[user])
            
            # 更新参数
			delta_rxi = score - prediction
			bu[user] += Lr * (delta_rxi - reg * bu[user])
			bi[item] += Lr * (delta_rxi - reg * bi[item])	
			for k in range(factorNum):
				temp = pu[user][k]	
				pu[user][k] += Lr * (delta_rxi * qi[item][k] - reg * pu[user][k])
				qi[item][k] += Lr * (delta_rxi * temp - reg * qi[item][k])
        
        #学习率衰减
		if Lr>0.1:
			Lr *= 0.9
		else:
			Lr = 0.05

        #输出RMSE
		curr_Rmse = validate(testData, aver, bu, bi, qi, pu)
		print("RMSE of epoch",epoch,": ", curr_Rmse)
		if curr_Rmse >= last_Rmse:
			break
		else:
			last_Rmse = curr_Rmse
		time2=datetime.datetime.now()
		run_time=(time2-time1).seconds
		print("Epoch",epoch," running time:  ",int(run_time/60),"min",run_time%60,"s")
	print("Training ends. RMSE is",curr_Rmse)
	test(aver, bu, bi, qi, pu)

#执行程序
SVD(testdata,traindata,10,0.95,0.2)
end = datetime.datetime.now()
run_time=(end-start).seconds
print("Running time :  ",int(run_time/60),"min",run_time%60,"s")