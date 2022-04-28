# timing_frame （简单择时框架）
### /timing_frame

#### tf_np -- np版择时框架
#### tf_pd -- pandas版择时框架

--------
data/里的数据库文件中有000001和600000两只股票，提供测试

function.py 包括框架所需函数
signals.py 主要生成signal(可以自定义函数，用于生成signal)

np版和pandas版结果会略有不同，主要原因是在计算时，保留小数位的不同造成的


-----------------------

### /select_stcok_frame 选股框架
例子为简单的市值因子，按月调仓


### 准备加入事件驱动框架


2022.04.08
author: guojunlei