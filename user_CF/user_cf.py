import multiprocessing
import numpy as np
import math


class UserCF:
    """
    user-user协同过滤
    """

    def __init__(self):
        """
        user id的最小值为0，最大值为19834
        item id的最小值为0，最大值为624960
        UserMatrix：第一维：user id；第二维：item id ：score 键值对
        item id ：score键值对 表示一个用户打分的所有电影和相应分数
        ItemMatrix：第一维：item id；第二维：user id ：score 键值对
        user id ：score键值对 表示一个电影中所有打分用户和其打分的分数
        AvgUserScore：所有用户的打分均分
        AvgItemScore：所有电影的均分
        ValidateData：验证集（自己划分进行rmse的测试）
        TestData：测试集
        """
        self.NumOfUser = 19835
        self.NumOfMovie = 624961
        self.UserMatrix = [dict() for i in range(self.NumOfUser)]
        self.ItemMatrix = [dict() for j in range(self.NumOfMovie)]
        # self.PearData = []
        self.AvgUserScore = []
        self.AvgItemScore = []
        self.ValidateData = []
        self.TestData = []

    def load_data(self):
        """
        加载训练集的数据
        格式：
            <user id>|<numbers of rating items>
            <item id>   <score>
        self.UserData以<user id>作为数组下标
        self.MovieData以<item id>作为数组下标
        """
        user_id = -1
        with open('data/train.txt', 'r', encoding='utf-8') as f:
            for line in f:
                if line.find('|') != -1:
                    user_id = line.split('|')[0]
                    user_id = int(user_id)
                else:
                    try:
                        item_id, item_score, _ = line.split('  ')
                        item_id, item_score = int(item_id), int(item_score)
                        self.UserMatrix[user_id][item_id] = item_score
                        self.ItemMatrix[item_id][user_id] = item_score
                    # 如果train.txt末尾有空行，则会报出此错误，这里选择忽略
                    except ValueError:
                        pass

    def user_avg_score(self, user_id):
        """
        计算一个用户所有打分的平均分
        :param user_id: 用户id
        :return: 一个用户的打分平均分
        """
        sum_of_scores = sum(self.UserMatrix[user_id].values())
        sum_of_items = len(self.UserMatrix[user_id])
        return sum_of_scores / sum_of_items

    def item_avg_score(self, item_id):
        """
        计算一个电影被打分的平均分
        若sum_of_users = 0 说明不存在该电影id（无用户给其打分） 返回-1即可
        :param item_id: 电影id
        :return: 一个电影被打分的平均分
        """
        sum_of_scores = sum(self.ItemMatrix[item_id].values())
        sum_of_users = len(self.ItemMatrix[item_id])
        try:
            return sum_of_scores / sum_of_users
        except ZeroDivisionError:
            return -1

    def all_avg_score(self):
        """
        计算所有分数的平均分
        :return:
        """
        sum_of_scores = 0
        count = 0
        for user_id in range(self.NumOfUser):
            sum_of_scores += sum(self.UserMatrix[user_id].values())
            count += len(self.UserMatrix[user_id])
        return sum_of_scores / count

    def compute_avg(self):
        """
        计算所有的均分并存入相应数组中
        方面后续计算皮尔森相关系数和baseline时用到
        """
        for i in range(self.NumOfUser):
            self.AvgUserScore.append(self.user_avg_score(i))
        for i in range(self.NumOfMovie):
            self.AvgItemScore.append(self.item_avg_score(i))

    def pearson_sim_xy(self, user_x, user_y):
        """
        计算用户x和用户y的pearson系数
        item_set: 用户x和用户y都看过的电影id集合
        score_x: r_xs - r_x 向量 （r_xs: 用户x给item_set中的电影s的打分；r_x：用户x的打分均分）
        score_y: r_ys - r_y 向量
        :param user_x: 用户x的id
        :param user_y: 用户y的id
        :return: pearson值
        """
        item_set = set(self.UserMatrix[user_x].keys()).intersection(set(self.UserMatrix[user_y].keys()))
        scores_x = np.array(list((v - self.AvgUserScore[user_x]) for k, v in self.UserMatrix[user_x].items()
                                 if k in item_set))
        scores_y = np.array(list((v - self.AvgUserScore[user_y]) for k, v in self.UserMatrix[user_y].items()
                                 if k in item_set))
        pear_numerator = np.sum(scores_x * scores_y)
        norm_scores_x = np.linalg.norm(scores_x)
        norm_scores_y = np.linalg.norm(scores_y)
        try:
            pearson = pear_numerator / (norm_scores_x * norm_scores_y)
        except ValueError:
            return 0
        return pearson

    def bxi(self, user_id, item_id, u):
        """
        compute baseline estimate for rxi
        bxi = bx + bi + u
        :param user_id: 用户id
        :param item_id: 电影id
        :param u: 所有电影打分均值
        :return: bxi
        """
        bx = self.AvgUserScore[user_id] - u
        bi = self.AvgItemScore[item_id] - u
        bxi = bx + bi + u
        return bxi

    def rrxi(self, i, data, per_count, u):
        result = []
        for user_id in range(per_count*i, min(per_count*(i+1), self.NumOfUser)):
            movie_scores = []
            if isinstance(data, dict):
                for movie_id in data[user_id].keys():
                    movie_scores.append(int(self.rxi(user_id, movie_id, u)))
            else:
                for movie_id in data[user_id]:
                    movie_scores.append(int(self.rxi(user_id, movie_id, u)))
            print(user_id, movie_scores)
            result.append(movie_scores)
        return result

    def rxi(self, user_id, item_id, u):
        """
        compute rxi
        rxi = bxi + sum(sij * (rxj - bxj)) / sum(sij)
        sij: 用户i与用户j的pearson相关系数
        这里只对与用户user_id观看过同一电影item_id且pearson相关系数大于0的用户i进行计算代入
        :param user_id: 用户id
        :param item_id: 电影id
        :param u: 所有电影打分均值
        :return: rxi
        """
        # print(user_id, item_id, u)
        sum_sij = 0
        sum_sij_rjx = 0
        for i in range(self.NumOfUser):
            if item_id in self.UserMatrix[i].keys() and i != user_id:
                a = self.pearson_sim_xy(user_id, i)
                if a > 0:
                    sum_sij += a
                    sum_sij_rjx += a * (self.UserMatrix[i][item_id] - self.bxi(i, item_id, u))

        if sum_sij != 0 and sum_sij_rjx != 0:
            rxi = sum_sij_rjx / sum_sij + self.bxi(user_id, item_id, u)
            rxi = max(0, min(100, rxi))
            return rxi
        else:
            return self.AvgUserScore[user_id]

    def get_test_data(self):
        """
        读入测试数据集，存入TestData中
        """
        user_id = -1
        with open('data/test.txt', 'r', encoding='utf-8') as f:
            for line in f:
                if line.find('|') != -1:
                    user_id = int(line.split('|')[0])
                    self.TestData.append([])
                else:
                    # 把初始值都设为0
                    self.TestData[user_id].append(int(line))

    def multi_process_test(self, data, u):
        """
        对测试集进行测试，结果存入result中
        :param u: 所有电影打分均值
        :return: result
        """
        cores = multiprocessing.cpu_count()
        result = [[] for i in range(cores)]
        # start a pool
        pool = multiprocessing.Pool(processes=cores)
        per_count = int(20000 / cores)
        for i in range(cores):
            print("begin process -- {}".format(i))
            result[i] = pool.apply_async(self.rrxi, args=(i, data, per_count, u))
        pool.close()
        pool.join()
        return result

    def divide_validation_data(self, n):
        """
        将UserData中的部分数据划分为验证集，存入ValidateData
        并删除UserData的对应项
        :param n: 每个用户的打分中当作验证集的个数
        """
        self.ValidateData.clear()
        for i in range(self.NumOfUser):
            val_movie_num = n
            count = 0
            self.ValidateData.append(dict())
            for item in self.UserMatrix[i].items():
                if count >= val_movie_num:
                    break
                self.ValidateData[i][item[0]] = item[1]
                count += 1

        # 删除Data中的对应项
        for i in range(self.NumOfUser):
            for j in self.ValidateData[i].keys():
                self.ItemMatrix[j].pop(i)
                self.UserMatrix[i].pop(j)

    def store_validation_data(self):
        """
        将划分为验证集的原数据进行存储
        validate.txt
        """
        with open('data/validate.txt', 'w', encoding='utf-8') as f:
            for i in range(self.NumOfUser):
                s = str(i) + '|' + str(len(self.ValidateData[i]))
                f.write(s + '\n')
                for key, value in self.ValidateData[i].items():
                    f.write(str(key) + '  ' + str(value) + '\n')

    def validation_test(self, num, u):
        """
        对验证集进行rmse测试
        :param u: 所有电影打分均值
        :return: rmse值
        """
        rmse = 0.0
        cores = multiprocessing.cpu_count()
        validate_result = self.multi_process_test(self.ValidateData, u)
        per_count = len(validate_result[0])
        for i in range(cores):
            for j in range(len(validate_result[i])):
                rmse += np.sum(pow(self.ValidateData[i*per_count+j].values() - validate_result[j], 2))
        print(math.sqrt(rmse / (self.NumOfUser * num)))
        return validate_result

    def write_test_res(self, file_name, num, data, result, u):
        """
        将测试结果一并写入文件
        :param file_name: 文件名
        :param num: 测试条数
        :param data:
        :param u: 所有电影打分均值
        """
        with open(file_name, 'w', encoding='utf-8') as f:
            for i in range(len(result)):
                for j in range(len(result[i])):
                    f.write("{}|{}\n".format(i, num))
                    count = 0
                    # 如果是验证集
                    if isinstance(data, dict):
                        for k in data[i*len(result[0])+j].keys():
                            f.write("{}  {}\n".format(k, result[i][j][count]))
                            count += 1
                    # 如果是测试集
                    else:
                        for k in range(num):
                            f.write("{}  {}\n".format(data[i*len(result[0])+j][k], result[i][j][k]))

    def validate(self):
        self.load_data()
        print('load finished')
        self.divide_validation_data(3)
        print('got validation data')
        self.compute_avg()
        u = self.all_avg_score()
        print('got avg data, begin testing')
        result = self.validation_test(3, u)
        self.write_test_res('data/res.txt', 3, self.ValidateData, result, u)
        print('succeed')

    def test(self):
        self.load_data()
        print('load finished')
        self.get_test_data()
        print('got test data')
        self.compute_avg()
        u = self.all_avg_score()
        print('got avg data, begin testing')
        result = self.multi_process_test(self.TestData, u)
        self.write_test_res('data/result.txt', 6, self.TestData, result, u)
        print('succeed')


if __name__ == '__main__':
    user_cf = UserCF()
    # user_cf.validate()
    user_cf.test()
