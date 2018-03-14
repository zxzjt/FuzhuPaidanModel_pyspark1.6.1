#! /use/bin/python3
# coding: utf8
import logging,sys,os
import subprocess,time
import csv

from pyspark import SQLContext,SparkContext,SparkConf
from pyspark.sql.types import *
from pyspark.ml.feature import OneHotEncoder,StringIndexer, IndexToString,VectorAssembler, VectorIndexer
from pyspark.ml.feature import MinMaxScaler,MinMaxScalerModel
from pyspark.sql import Row

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest,RandomForestModel
from pyspark.ml.classification import DecisionTreeClassifier, DecisionTreeClassificationModel
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.mllib.evaluation import MulticlassMetrics

class MultiLabelEncoder(object):
    """多标签编码器
    将多列字符型标称变量转化为整数编码
    参数
    features：list，所有字符字段名称
    属性
    encode_list：每个字段对应的一个LabelEncoder，所有的LabelEncoder存入该列表
    features：list，所有字符字段名称
    """
    def __init__(self,features):
        self.encode_list=[]
        self.features=features
        self.features_t = [item + '_t' for item in self.features]
        for i in range(len(features)):
            self.encode_list.append(StringIndexer(inputCol=self.features[i],outputCol=self.features[i]+'_t'))

    def fit_transform(self,X):
        """拟合转换
        对输入数据先拟合后转换编码
        :param X:输入的数据,DF
        :return:编码后的数据
        """
        self.encode_model_list = []
        temp_df = X
        cols_all = [ i for i in X.columns if i not in self.features]+self.features_t
        for i in range(len(self.encode_list)):
            self.encode_model_list.append(self.encode_list[i].fit(temp_df))
            temp_df = self.encode_model_list[i].transform(temp_df)
        return temp_df.select(cols_all)

    def transform(self,X):
        """转换
        对输入数据转换编码
        :param X:输入的数据,DF
        :return:编码后的数据
        """
        temp_df = X
        cols_all = [i for i in X.columns if i not in self.features] + self.features_t
        for i in range(len(self.encode_list)):
            temp_df = self.encode_model_list[i].transform(temp_df)
        return temp_df.select(cols_all)

class MultiOneHotEncoder(object):
    """多特征热编码器
    将多列数字特征转化为热编码（无需fit）
    参数
    features：list，待编码字段名称
    属性
    encode_list：每个字段对应的一个OneHotEncoder，所有的OneHotEncoder存入该列表
    features：list，待编码字段名称
    """
    def __init__(self,features):
        self.encode_list=[]
        self.features=features
        self.features_t = [item + '_t' for item in self.features]
        for i in range(len(features)):
            self.encode_list.append(OneHotEncoder(dropLast=False,inputCol=self.features[i],outputCol=self.features[i]+'_t'))

    def transform(self,X):
        """转换
        对输入数据转换编码
        :param X:输入的数据,DF
        :return:编码后的数据
        """
        temp_df = X
        cols_all = [i for i in X.columns if i not in self.features] + self.features_t
        for i in range(len(self.encode_list)):
            temp_df = self.encode_list[i].transform(temp_df)
        return temp_df.select(cols_all)

class DataChecker(object):
    """数据校验
    参数
    无
    属性
    类变量：
    std_data_values指标取值范围，或枚举
    keys_class,keys_num
    实例变量：
    item_num数据记录数
    feature_num数据字段数
    missing_keys缺失关键字段
    data_exception_keys数据异常的字段
    """
    # 使用的指标字段
    keys_class = [ '地市', '区县','网元要素', '数据来源','问题归类(一级)','问题归类(二级)',
                  '类别要素','处理优先级','目前状态','是否指纹库智能分析系统运算',
                  '是否质检通过','资管生命周期状态','业务要素','场景要素', '覆盖类型','覆盖场景']  #
    keys_num = ['告警触发次数', '日均流量(GB)']  # '中心经度', '中心维度'
    """新的原始问题库{问题归类(二级),主指标表征值}-->老库{问题现象,表征指标值}"""
    std_data_values = {'问题触发时间':[],
                       '地市':['杭州','宁波','温州','绍兴','嘉兴','湖州','丽水','金华','衢州','台州','舟山'],
                       '区县':['上城','下城','江干','拱墅','西湖','滨江','下沙','萧山','余杭','建德','富阳','临安','桐庐','淳安',
                             '海曙','江北','北仑','镇海','鄞州','奉化','余姚','慈溪','象山','宁海',
                             '鹿城','龙湾','瓯海','洞头','瑞安','乐清','永嘉','平阳','苍南','文成','泰顺',
                             '越城','绍兴','上虞','诸暨','嵊州','新昌',
                             '吴兴','南浔','德清','长兴','安吉',
                             '南湖','秀洲','海宁','平湖','桐乡','嘉善','海盐',
                             '婺城','金东','兰溪','东阳','永康','义乌','武义','浦江','磐安',
                             '柯城','衢江','江山','常山','开化','龙游',
                             '椒江','黄岩','路桥','临海','温岭','玉环','三门','天台','仙居',
                             '莲都','龙泉','青田','缙云','遂昌','松阳','云和','庆元','景宁','开发区',
                             '定海','普陀','岱山','嵊泗'],
                       '网络类型':['4G','2G'],
                       '网元要素':['基站','小区'],
                       '数据来源':['OTT智能定位平台', 'LTE-MR', 'SEQ', '北向性能'],
                       '问题归类(一级)':['干扰问题', '负荷问题', '结构问题', 'VOLTE问题', '性能问题', '覆盖问题', '感知问题', '互操作问题'],
                       '问题归类(二级)':['掉线质差', '接通质差', '切换质差', '语音MOS质差', '高负荷', 'SRVCC切换质差', '重叠覆盖',
                                       'VOLTE丢包质差', '上行SINR质量差', '弱覆盖', '过覆盖', 'CSFB性能质差', '零流量', '高干扰',
                                       'VOLTE接通质差', 'VOLTE掉话质差', '低速率'],
                       '问题类型':[],
                       '类别要素':['互操作','感知','质量','负荷','结构'],
                       '是否追加':['是','否'],
                       '主指标(事件)':[],
                       '主指标表征值':[-200,900],
                       '处理优先级':['中','高'],
                       '目前状态':['待接入','归档','人工关闭','已接入'],
                       '是否为FDD站点':['是','否'],
                       '是否实时工单已派单':['是','否'],
                       '是否指纹库智能分析系统运算':['是','否'],
                       '是否列为白名单':['是','否'],
                       '是否为性能交维站点':['是','否'],
                       '是否质检通过':['是','否','未质检'],
                       '资管生命周期状态':['现网','在网'],#'工程','维护','设计'
                       #'劣化次数':[1,31],
                       '告警触发次数':[1,500],
                       '日均流量(GB)':[0.0,1000],
                       '业务要素':['数据','语音'],
                       '触发要素':['劣于门限','异常事件','人工创造'],
                       '场景要素':['室分', '普铁', '风景区', '室外', '地铁', '高校', '山区', '全网', '高流量',
                                 '高速', '高铁', '美食街', '高层', '小微站', '海域'],
                       '覆盖类型':['室内','室外'],
                       '覆盖场景':['普铁', '地铁', '乡镇', '集贸市场', '公墓', '高速公路', '高层居民区', '写字楼', '高校',
                                 '低层居民区', '会展中心', '风景区', '边境小区', '中小学', '党政军机关', '国道省道',
                                 '党政军宿舍', '武警军区', '别墅群', '体育场馆', '郊区道路', '医院', '商业中心', '航道',
                                 '高铁', '企事业单位', '码头', '长途汽车站', '其他', '广场公园', '城区道路', '工业园区',
                                 '火车站', '机场', '休闲娱乐场所', '城中村', '村庄', '近水近海域', '山农牧林', '星级酒店'],
                       '二级场景':[],
                       '中心经度':[118.037,123.143],
                       '中心维度':[27.22,31.18],
                       'TAC(LAC)':[22148,26840]}

    def __init__(self):
        self.item_num = 0
        self.feature_num = 0
        self.missing_keys = []
        self.data_exception_keys = []

    def __null_process(self, data, nan_fill_data):
        """数据空值填充
        根据提供的fill_data填充空值数据
        :param data:待校验的数据
        :param nan_fill_data:各字段默认的填充值，dict
        :return:data_filled
        """
        data_filled = data.na.fill(nan_fill_data)
        return data_filled

    def __exception_process(self,row):
        data = []
        # 数值类型判断
        for key in DataChecker.keys_num:
            if (row[key] == '') or (float(row[key]) < DataChecker.std_data_values[key][0]) or (float(row[key]) > DataChecker.std_data_values[key][1]):
                data.append(None)
                self.exception_keys.add(key)
            else:
                data.append(float(row[key]))
        # 标称字段判断
        for key in DataChecker.keys_class:
            if row[key] == '' or row[key] not in DataChecker.std_data_values[key]:
                data.append(None)
                self.exception_keys.add(key)
            else:
                data.append(row[key])
        return data

    def data_check(self, data, sql_context, nan_fill_data):
        """数据校验
        根据预设的范围判断数据是否异常，先对数据空值填充，再校验
        :param data:待校验的数据
        :param sql_context:sql_context
        :param nan_fill_data:各字段默认的填充值
        :return:填充的数据
        """
        logger = logging.getLogger("ZiyuLogging")
        self.item_num = data.count()
        self.feature_num = len(data.columns)
        if self.item_num == 0 or self.feature_num == 0:
            # print("The file has no data!")
            logger.info("The file has no data!")
            self.no_data = '是'
            return (1,[])
        else:
            self.no_data = '否'
            for key in DataChecker.keys_num + DataChecker.keys_class:
                if key not in data.columns:
                    self.missing_keys.append(key)
                else:
                    pass
            if len(self.missing_keys) != 0:
                # print(self.missing_keys)
                logger.info('missing_keys: '+', '.join(self.missing_keys))
                return (2,[])
            else:
                self.missing_keys = []
                self.exception_keys = set()
                checked_rdd = data.rdd.map(lambda row: self.__exception_process(row))
                data_df_null = sql_context.createDataFrame(checked_rdd,DataChecker.keys_num+DataChecker.keys_class)
                data_filled = self.__null_process(data_df_null, nan_fill_data)
                if len(self.exception_keys) != 0:
                    logger.info('exception_keys: '+', '.join(self.exception_keys))
                return (0,data_filled)

class ZiyuDataPreProc(object):
    """工单自恢复分类器
    数据编码转换，模型训练，预测，优化
    参数
    model：输入可用的分类器模型
    属性
    实例变量：model,keys_class，keys_num，encoder1，encoder2，encoder3，encoder4
    """
    def __init__(self):
        # 数值归一化
        self.encoder1 = MinMaxScaler(inputCol='num_feats',outputCol='num_feats_t')
        # 输入标称字段编码成数字
        self.encoder2 = MultiLabelEncoder(DataChecker.keys_class)
        # 数字编码成one-hot
        self.encoder3 = MultiOneHotEncoder(self.encoder2.features_t)
        # 输出标签编码成数字
        self.encoder4 = StringIndexer(inputCol='自恢复状态',outputCol='label')
        pass

    def data_fit_transform(self,data):
        """数据拟合归一编码转换
        对输入数据拟合归一化，编码转换
        :param data:输入的数据data，包含y和X
        :return:返回归一编码后的数据
        """
        col_names = data.columns
        num_feats_out = [i for i in col_names if i not in DataChecker.keys_num] + ['num_feats']
        num_feats_t_out = [i for i in col_names if i not in DataChecker.keys_num] + ['num_feats_t']
        # 统计均值和众数
        self.mean_mode = self.__get_means_mode(data)
        # 数值字段
        num_feats_merger = VectorAssembler(inputCols=DataChecker.keys_num,outputCol='num_feats')
        num_merge_data = num_feats_merger.transform(data).select(num_feats_out)
        self.encoder1_model = self.encoder1.fit(num_merge_data)
        num_trans_data = self.encoder1_model.transform(num_merge_data).select(num_feats_t_out)
        # 名义字段编码
        class_trans1_data = self.encoder2.fit_transform(num_trans_data)
        class_trans2_data = self.encoder3.transform(class_trans1_data)
        # 标签编码
        self.encoder4_model = self.encoder4.fit(class_trans2_data)
        trans_data = self.encoder4_model.transform(class_trans2_data)
        # 合并features
        merged_feats = [self.encoder1.getOutputCol()]+self.encoder3.features_t
        feats_merger = VectorAssembler(inputCols=merged_feats, outputCol='features')
        trans_merge_data = feats_merger.transform(trans_data).select([i for i in trans_data.columns if i not in merged_feats]+['features'])
        return trans_merge_data

    def data_transform(self,X):
        """数据归一编码转换
        对输入数据归一化，编码转换
        :param X:输入的数据X
        :return:返回归一编码后的数据
        """
        feats = X.columns
        try:
            # 数值字段
            num_feats_out = [i for i in feats if i not in DataChecker.keys_num] + ['num_feats']
            num_feats_t_out = [i for i in feats if i not in DataChecker.keys_num] + ['num_feats_t']
            num_feats_merger = VectorAssembler(inputCols=DataChecker.keys_num, outputCol='num_feats')
            num_merge_data = num_feats_merger.transform(X).select(num_feats_out)
            num_trans_data = self.encoder1_model.transform(num_merge_data).select(num_feats_t_out)
            # 名义字段编码
            class_trans1_data = self.encoder2.transform(num_trans_data)
            class_trans2_data = self.encoder3.transform(class_trans1_data)
        except:
            logger = logging.getLogger("ZiyuLogging")
            logger.exception("data_transform错误")
            return []
        else:
            # 合并features
            merged_feats = [self.encoder1.getOutputCol()]+self.encoder3.features_t
            feats_merger = VectorAssembler(inputCols=merged_feats, outputCol='features')
            trans_merge_data = feats_merger.transform(class_trans2_data).select([i for i in class_trans2_data.columns if i not in merged_feats]+['features'])
            return trans_merge_data

    def __get_means_mode(self, data):
        """统计均值和众数
        :param X: 输入X
        :return: DataFrame,各字段均值或众数
        """
        # Mean
        summary = data.select(DataChecker.keys_num).describe().collect()
        mean_X = summary[1].asDict()
        mean_X.pop('summary')
        merged = {}
        for key in mean_X.keys():
            merged[key] = float(mean_X[key])
        # Mode
        mode_X = {}
        for key in DataChecker.keys_class:
            word_count = data.select(key).rdd.map(lambda x: (x[key], 1)).reduceByKey(lambda a, b: a + b)
            mode_x = dict(word_count.collect())
            mode_X[key] = max(mode_x, key=mode_x.get)
        # Merge
        merged.update(mode_X)
        return merged

class ZiyuLogging(object):
    """日志记录
    记录调试和校验日志
    """
    @staticmethod
    def config(logger,path):
        """日志配置
        :param logger:创建Logging对象
        :return:None
        """
        # 指定logger输出格式
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(filename)s:%(lineno)s - %(message)s')
        # 文件日志
        file_handler = logging.FileHandler(path+'ziyu_mode.log',encoding='utf8')
        file_handler.setFormatter(formatter)  # 可以通过setFormatter指定输出格式
        # 控制台日志
        #console_handler = logging.StreamHandler(sys.stdout)
        #console_handler.formatter = formatter  # 也可以直接给formatter赋值
        # 为logger添加的日志处理器
        logger.addHandler(file_handler)
        #logger.addHandler(console_handler)
        # 指定日志的最低输出级别，默认为WARN级别
        logger.setLevel(logging.INFO)

    def test_logging1(self):
        logger = logging.getLogger("ZiyuLogging")
        logger.info("Test ZiyuLogging")

    @classmethod
    def test_logging2(cls):
        cls.config(logger= logging.getLogger("OtherLogging"))
        logger = logging.getLogger("OtherLogging")
        logger.error("Test OtherLogging")

def data_trans(row_list):
    data = []
    for i in range(len(row_list)):
        if i == 12 or i == 13:
            data.append(float(row_list[i]))
        else:
            data.append(row_list[i])
    return data

def trans_to_csv(row,test_header):
    s = ','.join(str(row[key]) for key in test_header+["predictedLabel"])
    return s+'\n'

if __name__ == "__main__":
    #home_path = 'E:/MyPro/FuzhuPaidanModel_pyspark_test/'
    #home_path_local = 'E:/MyPro/FuzhuPaidanModel_pyspark_test/'
    home_path = '/user/znyw/zxzjt/FuzhuPaidanModel_pyspark1.6.1/'
    home_path_local = '/home/znyw/zhujingtao/FuzhuPaidanModel_pyspark1.6.1/'
    # 日志开启
    ZiyuLogging.config(logger=logging.getLogger("ZiyuLogging"), path=home_path_local)
    logger = logging.getLogger("ZiyuLogging")
    # 创建sc
    sc = SparkContext(appName="FuzhuPaidan")
    train_all = sc.textFile(home_path+'train.csv',use_unicode=True).map(lambda line: line.split(","))
    logger.info('train: sc.textFile')
    train_header = train_all.collect()[0]
    train_rdd = train_all.filter(lambda line: line[0] != train_header[0]).map(lambda row:data_trans(row))
    sqlContext = SQLContext(sc)
    train_df = sqlContext.createDataFrame(train_rdd,train_header)
    logger.info('train: sqlContext.createDataFrame')
    model_path = home_path+"model"
    #data_all = sqlContext.read.format("com.databricks.spark.csv").option('header',True).load(path='./train.txt')
    train = train_df
    pre_proc = ZiyuDataPreProc()
    train_proc = pre_proc.data_fit_transform(train)
    logger.info('train: pre_proc.data_fit_transform')
    #rf = RandomForestClassifier(labelCol="label", featuresCol="features", maxDepth=10,numTrees=100,featureSubsetStrategy='onethird',seed=10)
    #model = rf.fit(train_proc)
    #model.write().overwrite().save(model_path)
    train_proc_label_point = train_proc.rdd.map(lambda row:LabeledPoint(row['label'],row['features']))
    logger.info('train: generate train_proc_label_point')
    model = RandomForest.trainClassifier(train_proc_label_point, numClasses=2, categoricalFeaturesInfo={},numTrees=100, featureSubsetStrategy="onethird",impurity='gini', maxDepth=8,seed=20)
    logger.info('train: trainClassifier')
    #model.save(sc,model_path)
    logger.info('train: model.save')

    ## 实际部署
    # 创建校验
    data_checker = DataChecker()
    nan_fill_data = pre_proc.mean_mode
    # 数据目录
    data_dir = home_path+'test_dir/'
    res_dir = home_path+'res/'
    cmd1 = 'hdfs dfs -ls -R ' + data_dir
    # 加载模型
    #model_load = RandomForestClassificationModel.load(model_path)
    model_load = model #RandomForestModel.load(sc,model_path)
    logger.info('train: model.load')
    while True:
        files_list = subprocess.check_output(cmd1.split()).strip().split(b'\n')#shell=True for win 7, and '\r\n'
        if len(files_list) == 0:
            pass
        else:
            csv_files = [x for x in files_list if x.endswith(b".csv")]
            if len(csv_files) == 0:
                pass
            else:
                for file in csv_files:
                    start_time = time.time()
                    res_list = []
                    file_name = file.split(b'/')[-1].decode('utf-8')
                    logger.info('test: get csv file, %s' % file_name)
                    # test_data = spark.read.csv(path=data_dir + file_name, encoding='gbk', header=True, inferSchema=True)
                    test_all = sc.textFile(data_dir + file_name,use_unicode=True).map(lambda line: line.split(","))
                    logger.info('test: sc.textFile, %s' % file_name)
                    test_header = test_all.first()
                    test_rdd = test_all.filter(lambda line: line[0] != test_header[0])
                    test_data = sqlContext.createDataFrame(test_rdd,test_header)
                    logger.info('test: sqlContext.createDataFrame, %s' % file_name)
                    test_data_ava = test_data.select(DataChecker.keys_num+DataChecker.keys_class)
                    # 添加简单校验规则
                    data_status = data_checker.data_check(test_data_ava, sqlContext, nan_fill_data)
                    logger.info('test: data_checker.data_check, %s' % file_name)
                    if data_status[0] != 0:
                        logger.info('test: data_status[0] is %s exception, %s' % (str(data_status[0]),file_name))
                    else:
                        trans_test = pre_proc.data_transform(data_status[1])
                        if trans_test == []:
                            logger.info('test: pre_proc.data_transform exception, %s' % file_name)
                            pass
                        else:
                            trans_test_label_point = trans_test.rdd.map(lambda row: row['features'])
                            logger.info('test: generate trans_test_label_point, %s' % file_name)
                            predicter_collect = model_load.predict(trans_test_label_point).collect()
                            logger.info('test: model_load.predict, %s' % file_name)
                            test_data_collect = test_data.collect()
                            test_data_cols = test_data.columns
                            for i in range(test_data.count()):
                                test_data_collect_i = test_data_collect[i]
                                row = [test_data_collect_i[key] for key in test_data_cols]
                                row.append(predicter_collect[i])
                                res_list.append(row)
                            res_df = sqlContext.createDataFrame(res_list,test_data_cols+["prediction"])
                            logger.info('test: test_data append prediction column , %s' % file_name)
                            labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",labels=pre_proc.encoder4_model.labels)
                            res = labelConverter.transform(res_df)
                            logger.info('test: prediction column IndexToString, %s' % file_name)
                            res.limit(5).show()
                            # 使用spark写parquet文件
                            #res_named = res.withColumnRenamed('问题归类(一级)', '问题归类_一级').withColumnRenamed('问题归类(二级)', '问题归类_二级').withColumnRenamed('日均流量(GB)', '日均流量_GB')
                            #res_named.write.parquet(path=res_dir+file_name+'.res',mode='overwrite')
                            # trans_to_csv
                            res.rdd.repartition(1).map(lambda row:trans_to_csv(row,test_header)).saveAsTextFile(res_dir+file_name+'.res')
                            logger.info('test: result output to csv, %s' % file_name)
                            cmd2 = 'hdfs dfs -mv '+data_dir+file_name+ ' '+data_dir+file_name+'.back'
                            subprocess.check_output(cmd2.split())#shell=True for win 7
                            logger.info('test: rename origin .csv file to .back, %s' % file_name)
                            time.sleep(1)
                    end_time = time.time()
                    logger.info('predition takes %s s, %s' % (str(end_time-start_time),file_name))
                            # train_pred = model_load.transform(train_proc).select(['prediction','label']).rdd.map(lambda row:(row['prediction'],row['label']))
                            # metrics = MulticlassMetrics(train_pred)
                            # print(metrics.confusionMatrix().toArray())
                time.sleep(10)
                pass