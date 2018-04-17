#! /use/bin/python
# coding: utf8
import logging,sys,os,ftplib
import subprocess,time
import pandas as pd

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

"""python 2 编码总结
sc.textFile读入utf8文件，设置use_unicode=True，用sqlContext.createDateFrame后字段名columns为utf8的str类型，取值为unicode类型
用select时使用utf8或unicode作字段名都可以，collect后的Row类型只能使用utf8取其value
vector元素为np.float64类型，需用float显式转换后才能转到DoubleType
"""

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
    id = [u'问题点编号']
    id_str = [key.encode('utf8') for key in id]
    keys_class = [ u'地市', u'区县',u'网元要素', u'数据来源',u'问题归类(一级)',u'问题归类(二级)',
                  u'类别要素',u'处理优先级',u'目前状态',u'是否指纹库智能分析系统运算',
                  u'是否质检通过',u'资管生命周期状态',u'业务要素',u'场景要素', u'覆盖类型',u'覆盖场景']  #
    keys_num = [u'告警触发次数', u'日均流量(GB)']  # '中心经度', '中心维度'
    keys_class_str =[key.encode('utf8') for key in keys_class]
    keys_num_str = [key.encode('utf8') for key in keys_num]
    # schema
    data_types = StructType([StructField(id[0], StringType(), True),
                             StructField(u'告警触发次数', DoubleType(), True), StructField(u'日均流量(GB)', DoubleType(), True),
                             StructField(u'地市', StringType(), True), StructField(u'区县', StringType(), True),
                             StructField(u'网元要素', StringType(), True), StructField(u'数据来源', StringType(), True),
                             StructField(u'问题归类(一级)', StringType(), True), StructField(u'问题归类(二级)', StringType(), True),
                             StructField(u'类别要素', StringType(), True), StructField(u'处理优先级', StringType(), True),
                             StructField(u'目前状态', StringType(), True), StructField(u'是否指纹库智能分析系统运算', StringType(), True),
                             StructField(u'是否质检通过', StringType(), True), StructField(u'资管生命周期状态', StringType(), True),
                             StructField(u'业务要素', StringType(), True),StructField(u'场景要素', StringType(), True),
                             StructField(u'覆盖类型', StringType(), True),StructField(u'覆盖场景', StringType(), True)])
    """新的原始问题库{问题归类(二级),主指标表征值}-->老库{问题现象,表征指标值}"""
    std_data_values = {'问题触发时间':[],
                       '地市':[u'杭州',u'宁波',u'温州',u'绍兴',u'嘉兴',u'湖州',u'丽水',u'金华',u'衢州',u'台州',u'舟山'],
                       '区县':[u'上城',u'下城',u'江干',u'拱墅',u'西湖',u'滨江',u'下沙',u'萧山',u'余杭',u'建德',u'富阳',u'临安',u'桐庐',u'淳安',
                             u'海曙',u'江北',u'北仑',u'镇海',u'鄞州',u'奉化',u'余姚',u'慈溪',u'象山',u'宁海',
                             u'鹿城',u'龙湾',u'瓯海',u'洞头',u'瑞安',u'乐清',u'永嘉',u'平阳',u'苍南',u'文成',u'泰顺',
                             u'越城',u'绍兴',u'上虞',u'诸暨',u'嵊州',u'新昌',
                             u'吴兴',u'南浔',u'德清',u'长兴',u'安吉',
                             u'南湖',u'秀洲',u'海宁',u'平湖',u'桐乡',u'嘉善',u'海盐',
                             u'婺城',u'金东',u'兰溪',u'东阳',u'永康',u'义乌',u'武义',u'浦江',u'磐安',
                             u'柯城',u'衢江',u'江山',u'常山',u'开化',u'龙游',
                             u'椒江',u'黄岩',u'路桥',u'临海',u'温岭',u'玉环',u'三门',u'天台',u'仙居',
                             u'莲都',u'龙泉',u'青田',u'缙云',u'遂昌',u'松阳',u'云和',u'庆元',u'景宁',u'开发区',
                             u'定海',u'普陀',u'岱山',u'嵊泗'],
                       '网络类型':[u'4G',u'2G'],
                       '网元要素':[u'基站',u'小区'],
                       '数据来源':[u'OTT智能定位平台', u'LTE-MR', u'SEQ', u'北向性能'],
                       '问题归类(一级)':[u'干扰问题', u'负荷问题', u'结构问题', u'VOLTE问题', u'性能问题', u'覆盖问题', u'感知问题', u'互操作问题'],
                       '问题归类(二级)':[u'掉线质差', u'接通质差', u'切换质差', u'语音MOS质差', u'高负荷', u'SRVCC切换质差', u'重叠覆盖',
                                       u'VOLTE丢包质差', u'上行SINR质量差', u'弱覆盖', u'过覆盖', u'CSFB性能质差', u'零流量', u'高干扰',
                                       u'VOLTE接通质差', u'VOLTE掉话质差', u'低速率'],
                       '问题类型':[],
                       '类别要素':[u'互操作',u'感知',u'质量',u'负荷',u'结构'],
                       '是否追加':[u'是',u'否'],
                       '主指标(事件)':[],
                       '主指标表征值':[-200,900],
                       '处理优先级':[u'中',u'高'],
                       '目前状态':[u'待接入',u'归档',u'人工关闭',u'已接入'],
                       '是否为FDD站点':[u'是',u'否'],
                       '是否实时工单已派单':[u'是',u'否'],
                       '是否指纹库智能分析系统运算':[u'是',u'否'],
                       '是否列为白名单':[u'是',u'否'],
                       '是否为性能交维站点':[u'是',u'否'],
                       '是否质检通过':[u'是',u'否',u'未质检'],
                       '资管生命周期状态':[u'现网',u'在网'],#'工程','维护','设计'
                       #'劣化次数':[1,31],
                       '告警触发次数':[1,500],
                       '日均流量(GB)':[0.0,1000],
                       '业务要素':[u'数据',u'语音'],
                       '触发要素':[u'劣于门限',u'异常事件',u'人工创造'],
                       '场景要素':[u'室分', u'普铁', u'风景区', u'室外', u'地铁', u'高校', u'山区', u'全网', u'高流量',
                                 u'高速', u'高铁', u'美食街', u'高层', u'小微站', u'海域'],
                       '覆盖类型':[u'室内',u'室外'],
                       '覆盖场景':[u'普铁', u'地铁', u'乡镇', u'集贸市场', u'公墓', u'高速公路', u'高层居民区', u'写字楼', u'高校',
                                 u'低层居民区', u'会展中心', u'风景区', u'边境小区', u'中小学', u'党政军机关', u'国道省道',
                                 u'党政军宿舍', u'武警军区', u'别墅群', u'体育场馆', u'郊区道路', u'医院', u'商业中心', u'航道',
                                 u'高铁', u'企事业单位', u'码头', u'长途汽车站', u'其他', u'广场公园', u'城区道路', u'工业园区',
                                 u'火车站', u'机场', u'休闲娱乐场所', u'城中村', u'村庄', u'近水近海域', u'山农牧林', u'星级酒店'],
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
        for i_key in DataChecker.id_str:
            data.append(row[i_key])
        # 数值类型判断
        for key in DataChecker.keys_num_str:
            if (row[key] == '') or (float(row[key]) < DataChecker.std_data_values[key][0]) or (float(row[key]) > DataChecker.std_data_values[key][1]):
                data.append(None)
                #self.exception_keys.add(key)
            else:
                data.append(float(row[key]))
        # 标称字段判断
        for key in DataChecker.keys_class_str:
            if row[key] == '' or row[key] not in DataChecker.std_data_values[key]:
                data.append(None)
                #self.exception_keys.add(key)
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
            self.no_data = u'是'
            return (1,[])
        else:
            self.no_data = u'否'
            logger.info("data_check: item_num %s, feature_num %s" % (self.item_num,self.feature_num))
            #logger.info("data_check: column names %s" % data.columns)
            for key in DataChecker.keys_num_str + DataChecker.keys_class_str:
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
                #self.exception_keys = set()
                #data.limit(5).show()
                checked_rdd = data.rdd.map(lambda row: self.__exception_process(row))
                data_df_null = sql_context.createDataFrame(checked_rdd,create_schema(DataChecker.id+DataChecker.keys_num+DataChecker.keys_class))
                #data_df_null.limit(5).show()
                data_filled = self.__null_process(data_df_null, nan_fill_data)
                #data_filled.limit(5).show()
                #if len(self.exception_keys) != 0:
                #    logger.info('exception_keys: '+', '.join(self.exception_keys))
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
        self.encoder2 = MultiLabelEncoder(DataChecker.keys_class_str)
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
        logger = logging.getLogger("ZiyuLogging")
        col_names = data.columns# str type
        logger.info("Before ZiyuDataPreProc.data_fit_transform: item_num %s, feature_num %s" % (data.count(),len(col_names)))
        #logger.info("Before ZiyuDataPreProc.data_fit_transform: column names %s" % col_names)
        num_feats_out = [i for i in col_names if i not in DataChecker.keys_num_str] + ['num_feats']
        num_feats_t_out = [i for i in col_names if i not in DataChecker.keys_num_str] + ['num_feats_t']
        # 统计均值和众数
        self.mean_mode = self.__get_means_mode(data)# name is str type
        logger.info("ZiyuDataPreProc.data_fit_transform: mean_mode %s" % self.mean_mode)
        # 数值字段
        num_feats_merger = VectorAssembler(inputCols=DataChecker.keys_num_str,outputCol='num_feats')
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
        logger = logging.getLogger("ZiyuLogging")
        feats = X.columns
        logger.info("Before ZiyuDataPreProc.data_transform: item_num %s, feature_num %s" % (X.count(), len(feats)))
        #logger.info("Before ZiyuDataPreProc.data_transform: column names %s" % feats)
        try:
            # 数值字段
            num_feats_out = [i for i in feats if i not in DataChecker.keys_num_str] + ['num_feats']
            num_feats_t_out = [i for i in feats if i not in DataChecker.keys_num_str] + ['num_feats_t']
            num_feats_merger = VectorAssembler(inputCols=DataChecker.keys_num_str, outputCol='num_feats')
            num_merge_data = num_feats_merger.transform(X).select(num_feats_out)
            num_trans_data = self.encoder1_model.transform(num_merge_data).select(num_feats_t_out)
            # 名义字段编码
            class_trans1_data = self.encoder2.transform(num_trans_data)
            class_trans2_data = self.encoder3.transform(class_trans1_data)
        except:
            logger.exception("ZiyuDataPreProc.data_transform error")
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
        # Mean，Dataframe字段名为utf8 str，select可用utf8或unicode来查找
        summary = data.select(DataChecker.keys_num_str).describe().collect()
        mean_X = summary[1].asDict()
        mean_X.pop('summary')
        merged = {}
        for key in mean_X.keys():
            merged[key] = float(mean_X[key])
        # Mode，row取值为unicode，字段名为utf8来检索row[u'中文'.encode('utf8')]
        # Dataframe字段名为utf8 str，select可用utf8或unicode来查找
        mode_X = {}
        for key in DataChecker.keys_class_str:
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
    def test_logging2(cls,path):
        cls.config(logger= logging.getLogger("OtherLogging"),path=path)
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
    row_new=[row[key] for key in test_header+["predictedLabel"]]
    s = ','.join(row_new)
    return s+'\n'

def create_schema_test(header):
    data_schema = StructType()
    for key in header:
        data_schema.add(StructField(key.encode('utf8'), StringType(), True))
    return data_schema


def create_schema(header):
    data_schema = StructType()
    for key in header:
        if key in DataChecker.keys_num:
            data_schema.add(StructField(key.encode('utf8'), DoubleType(), True))
        else:
            data_schema.add(StructField(key.encode('utf8'), StringType(), True))
    return data_schema

class FtpTransmit(object):
    """ftp传输模块
    将结果传输到指定ftp目录

    """
    def __init__(self):
        self.host = '10.78.138.124'
        self.username = 'ch_etl'
        self.password = 'Ch_Etl1543!'

    def ftp_put(self, file_local, file_ftp):
        """
        ftp上传
        :param file_local:本地源文件
        :param file_ftp:ftp目标文件
        :return:
        """
        logger = logging.getLogger("ZiyuLogging")
        try:
            self.f = ftplib.FTP(self.host)  # 实例化FTP对象
        except:
            logger.exception("ftp error: cannot reach %s" % self.host)
        try:
            self.f.login(self.username, self.password)  # 登录
        except:
            logger.exception("ftp error: ftp login when ftp put, %s" % file_local)
        try:
            fp = open(file_local, 'r')
            self.f.storbinary('STOR ' + file_ftp, fp, 1024)
            fp.close()
        except:
            logger.exception("ftp error: ftp storbinary, %s" % file_local)
        self.f.quit()

if __name__ == "__main__":
    # python 2 重设默认编码
    #reload(sys)
    #sys.setdefaultencoding('utf8')
    #home_path = 'E:/MyPro/FuzhuPaidanModel_pyspark_test/'
    #home_path_local = 'E:/MyPro/FuzhuPaidanModel_pyspark_test/'
    home_path = '/user/rc_znpd/zxzjt/FuzhuPaidanModel_pyspark1.6.1/'
    home_path_local = '/home/rc_znpd/zxzjt/FuzhuPaidanModel_pyspark1.6.1/'
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
    train_df = sqlContext.createDataFrame(train_rdd,create_schema(train_header))
    logger.info('train: sqlContext.createDataFrame')
    model_path = home_path+"model"
    #data_all = sqlContext.read.format("com.databricks.spark.csv").option('header',True).load(path='./train.txt')
    train = train_df
    pre_proc = ZiyuDataPreProc()
    train_proc = pre_proc.data_fit_transform(train)
    logger.info('train: pre_proc.data_fit_transform')

    # use ml
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", maxDepth=8,numTrees=100,featureSubsetStrategy='onethird',seed=10)
    model = rf.fit(train_proc)
    logger.info('train: rf.fit')
    #model.write().overwrite().save(model_path)
    logger.info('train: model.save')

    """
    # use mllib
    train_proc_label_point = train_proc.rdd.map(lambda row:LabeledPoint(row['label'],row['features']))
    logger.info('train: generate train_proc_label_point')
    model = RandomForest.trainClassifier(train_proc_label_point, numClasses=2, categoricalFeaturesInfo={},numTrees=100, featureSubsetStrategy="onethird",impurity='gini', maxDepth=8,seed=20)
    logger.info('train: trainClassifier')
    #model.save(sc,model_path)
    logger.info('train: model.save')
    """
    ## 实际部署
    # 创建校验
    data_checker = DataChecker()
    nan_fill_data = pre_proc.mean_mode
    ftp_transer = FtpTransmit()
    # 数据目录
    ftp_res_dir = '/opt/znyw/result_data/'
    streamsets_dir = '/user/rc_znpd/zxzjt/wenti_data/'
    data_dir = home_path+'test_dir/'
    back_dir = home_path+'back_dir/'
    res_dir = home_path+'res/'
    res_dir_local = home_path_local+'res/'
    cmd1 = 'hdfs dfs -ls -R ' + data_dir
    cmd3 = 'hdfs dfs -ls -R ' + streamsets_dir
    # 加载模型
    #model_load = RandomForestClassificationModel.load(model_path)
    model_load = model #RandomForestModel.load(sc,model_path)
    logger.info('train: model.load')
    while True:
        # files transmission
        try:
            files_list_trans = subprocess.check_output(cmd3.split(), shell=False).strip().split(b'\n')
        except:
            logger.info('cmd3 error')
        else:
            files_list_trans = [file for file in files_list_trans if not file.startswith(b'd')]
            if len(files_list_trans) == 0:
                time.sleep(2)
                pass
            else:
                for file in files_list_trans:
                    file_name_trans = file.split(b'/')[-1]
                    cmd4 = 'hdfs dfs -mv ' + streamsets_dir + file_name_trans + ' ' + data_dir + file_name_trans + '.csv'
                    try:
                        subprocess.check_output(cmd4.split(),shell=False)  # shell=True for win 7
                    except:
                        logger.info('cmd4 hdfs dfs -mv error, %s' % file_name_trans)
                    else:
                        logger.info('test: cmd4 hdfs dfs -mv to .csv, %s' % file_name_trans)
                        time.sleep(2)
                        pass
        # process test_dir files
        files_list = subprocess.check_output(cmd1.split(),shell=False).strip().split(b'\n')#shell=True for win 7, and '\r\n'
        if len(files_list) == 0:
            time.sleep(2)
            pass
        else:
            csv_files = [x for x in files_list if x.endswith(b".csv")]
            if len(csv_files) == 0:
                time.sleep(2)
                pass
            else:
                for file in csv_files:
                    start_time = time.time()
                    res_list = []
                    file_name = file.split(b'/')[-1]#.decode('utf-8')
                    logger.info('test: get csv file, %s' % file_name)
                    # test_data = spark.read.csv(path=data_dir + file_name, encoding='gbk', header=True, inferSchema=True)
                    test_all = sc.textFile(data_dir + file_name,use_unicode=True).map(lambda line: line.split(","))
                    logger.info('test: sc.textFile, %s' % file_name)
                    test_header = test_all.first()
                    #logger.info('test: test_header is %s' % test_header)
                    test_rdd = test_all.filter(lambda line: line[0] != test_header[0])
                    #logger.info('test: the fist row of test_rdd is %s' % test_rdd.collect()[0])
                    test_data = sqlContext.createDataFrame(test_rdd,create_schema_test(test_header))
                    logger.info('test: sqlContext.createDataFrame, %s' % file_name)
                    #logger.info('test: columns of test_data are %s' % test_data.columns)
                    logger.info('test: test_data item_num %s, feature_num %s' % (test_data.count(), len(test_data.columns)))
                    test_data_ava = test_data.select(DataChecker.id_str+DataChecker.keys_num_str+DataChecker.keys_class_str)
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
                            """
                            # use mllib
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
                            """

                            # use ml
                            predicter = model_load.transform(trans_test)
                            logger.info('test: model_load.transform, %s' % file_name)
                            labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",labels=pre_proc.encoder4_model.labels)
                            res_rdd = labelConverter.transform(predicter).select(DataChecker.id_str+["prediction", "predictedLabel", 'probability']).map(lambda row:[row[DataChecker.id_str[0]],row["prediction"],row["predictedLabel"],float(row['probability'][1])])
                            res= sqlContext.createDataFrame(res_rdd,StructType([StructField(DataChecker.id_str[0],StringType(),True),StructField("prediction",DoubleType(),True),
                                                                               StructField("predictedLabel",StringType(),True), StructField('probability',DoubleType(),True)]))
                            #res.limit(5).show()
                            logger.info('test: prediction column IndexToString, %s' % file_name)
                            logger.info('test: prediction column IndexToString labels, %s' % pre_proc.encoder4_model.labels)
                            #logger.info('test: columns of res are %s' % res.columns)
                            logger.info('test: res item_num %s, feature_num %s' % (res.count(),len(res.columns)))
                            join_columns = [test_data[key] for key in test_data.columns]+[res["prediction"],res["predictedLabel"],res["probability"]]
                            res_join = test_data.join(res,test_data[DataChecker.id_str[0]]==res[DataChecker.id_str[0]],'left_outer').select(join_columns)
                            #logger.info('test: columns of res_join are %s' % res_join.columns)
                            logger.info('test: res_join item_num %s, feature_num %s' % (res_join.count(), len(res_join.columns)))
                            #res_join.limit(5).show()
                            # 使用spark写parquet文件
                            #res_named = res.withColumnRenamed('问题归类(一级)', '问题归类_一级').withColumnRenamed('问题归类(二级)', '问题归类_二级').withColumnRenamed('日均流量(GB)', '日均流量_GB')
                            #res_named.write.parquet(path=res_dir+file_name+'.res',mode='overwrite')
                            # trans_to_csv
                            #res.rdd.repartition(1).map(lambda row:trans_to_csv(row,test_header)).saveAsTextFile(res_dir+file_name+'.res')
                            res_join.repartition(1).write.json(res_dir+file_name+'.res','overwrite')
                            res_join.repartition(1).toPandas().to_csv(path_or_buf=res_dir_local + file_name + '.res.csv',
                                                     sep=',', encoding='utf8', index=False)
                            logger.info('test: result output, %s' % file_name)
                            time.sleep(3)
                            ftp_transer.ftp_put(res_dir_local + file_name + '.res.csv',ftp_res_dir + file_name + '.res.csv')
                            logger.info('test: finish result ftp_put, %s' % file_name)
                            cmd2 = 'hdfs dfs -mv '+data_dir+file_name+ ' '+back_dir+file_name+'.back'
                            try:
                                subprocess.check_output(cmd2.split(),shell=False)#shell=True for win 7
                            except:
                                logger.info('cmd2 hdfs dfs -mv error, %s' % file_name)
                            else:
                                logger.info('test: rename origin .csv file to .back, %s' % file_name)
                                time.sleep(2)
                    end_time = time.time()
                    logger.info('predition takes %s s, %s' % (str(end_time-start_time),file_name))
                            # train_pred = model_load.transform(train_proc).select(['prediction','label']).rdd.map(lambda row:(row['prediction'],row['label']))
                            # metrics = MulticlassMetrics(train_pred)
                            # print(metrics.confusionMatrix().toArray())
                time.sleep(10)
                pass