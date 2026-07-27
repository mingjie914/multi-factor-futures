import os

import dolphindb as ddb
import pandas as pd

# Standalone callers use the same environment contract as core.config.
_host = os.environ.get("MF_DDB_HOST", "")
hosts = [_host] if _host else []
ports = [int(os.environ.get("MF_DDB_PORT", "8961"))]
user = os.environ.get("MF_DDB_USER", "")
password = os.environ.get("MF_DDB_PASSWORD", "")

# ============================================================================
# 常用数据库与表路径常量（避免在各业务脚本中硬编码 dfs 路径）
# 详细字段说明请参阅《DolphinDB数据库完整检查报告.md》
# ============================================================================

# --- 数据库路径 ---
DB_GUOSEN_L2 = 'dfs://guosen_snapshot_level2_history_test'   # 国信证券 Level2 历史快照库（每日更新）
DB_ALPHA = 'dfs://alpha_db'                                   # Alpha 因子库（每日更新）
DB_KLINE = 'dfs://kline_db'                                   # 1 秒级 K 线库
DB_GUOSEN_BOND = 'dfs://guosen_bond_broker_best_history'      # 国信债券中介报价历史库
DB_BOND_BIAS = 'dfs://bond_db'                                # 债券偏离值库
DB_SWAP = 'dfs://swap'                                        # 互换业务库（2023-04 停更，仅供历史回测）
DB_WIND = 'dfs://wind_db'                                     # Wind 金融数据库（2023-05 停更，仅供历史回测）
DB_MYDB = 'dfs://mydb'                                        # Level2 逐笔行情库（2022-12 停更）
DB_TEST = 'dfs://test'                                        # 测试库

# --- 国信 Level2 快照表 ---
TBL_STOCK_L2_FULL = 'guosen_stock_snapshot_table'             # 股票 Level2 完整快照（108.8 亿行）
TBL_STOCK_L2_DEFAULT = 'guosen_stock_snapshot_default_table'  # 股票 Level2 默认快照（99.3 亿行）
TBL_FUND_L2 = 'guosen_fund_l2_snapshot_table'                 # 基金 Level2 快照（23.2 亿行，含 IOPV）

# --- Alpha 因子表 ---
TBL_DAILY_ALPHA = 'daily_alpha_table'                         # 每日 Alpha 因子值（8.4 亿行）

# --- K 线表（1 秒级） ---
TBL_KLINE_SSE = 'kline_sse_1sec'                              # 上交所 1 秒 K 线
TBL_KLINE_SZSE = 'kline_szse_1sec'                            # 深交所 1 秒 K 线
TBL_KLINE_CFE = 'kline_cfe_1sec'                              # 中金所 1 秒 K 线
TBL_KLINE_BSE = 'kline_bse_1sec'                              # 北交所 1 秒 K 线（空表）
TBL_KLINE_INDEX = 'kline_index_1sec'                          # 指数 1 秒 K 线（空表）

# --- 国信债券中介报价表 ---
TBL_BOND_BROKER = 'BondBrokerBest'                            # CFETS 债券中介报价（1.07 亿行）

# --- Wind 期货日行情表 ---
TBL_BOND_FUT_EOD = 'CBondFuturesEODPrices'                    # 债券期货日行情
TBL_INDEX_FUT_EOD = 'CIndexFuturesEODPrices'                  # 股指期货日行情
TBL_COMM_FUT_EOD = 'CCommodityFuturesEODPrices'               # 商品期货日行情

# --- Wind A 股日行情表 ---
TBL_ASHARE_EOD = 'AShareEODPrices'                            # A 股日行情
TBL_AINDEX_EOD = 'AIndexEODPrices'                            # 指数日行情


def get_Conn(host=None, port=None):
    '''
    获取 DolphinDB 会话连接，自动尝试配置的地址组合
    conn = get_Conn()                            # 自动尝试所有 host:port 组合
    conn = get_Conn('172.24.128.112', 8961)      # 指定地址和端口
    使用完毕请 conn.close()
    :param host: 主机地址，默认尝试 hosts 列表
    :param port: 端口，默认尝试 ports 列表
    :return: DolphinDB Session 对象，连接失败返回 None
    '''
    target_hosts = [host] if host else hosts
    target_ports = [port] if port else ports
    for h in target_hosts:
        for p in target_ports:
            s = ddb.Session()
            try:
                s.connect(h, p, user, password)
                return s
            except Exception as e:
                print(f"连接 {h}:{p} 失败: {e}")
                s.close()
    return None


def getDataCon(sqlstr, session):
    '''
    执行 DolphinDB 查询脚本，返回 DataFrame
    sqlstr = 'select * from loadTable("dfs://valuedb", "tb1")'
    getDataCon(sqlstr, session)
    执行完函数请 session.close()
    :param sqlstr: DolphinDB 脚本语句
    :param session: DolphinDB 会话连接
    :return: 查询结果 DataFrame，查询失败返回空 DataFrame
    '''
    try:
        df_data = session.run(sqlstr)
        if isinstance(df_data, pd.DataFrame):
            return df_data
        # 非表结果（标量、向量、字典等）统一封装为单列 DataFrame，保持返回类型一致
        return pd.DataFrame({'result': [df_data]})
    except Exception as e:
        print(f"查询出错: {e}")
        return pd.DataFrame()


def fetchData(sqlstr, host=None, port=None):
    '''
    一次性查询：自动建立连接、执行查询、关闭连接，适合单次取数场景
    sqlstr = 'select * from loadTable("dfs://valuedb", "tb1")'
    df = fetchData(sqlstr)
    :param sqlstr: DolphinDB 脚本语句
    :param host: 主机地址，默认尝试 hosts 列表
    :param port: 端口，默认尝试 ports 列表
    :return: 查询结果 DataFrame，失败返回空 DataFrame
    '''
    session = get_Conn(host, port)
    if session is None:
        return pd.DataFrame()
    try:
        return getDataCon(sqlstr, session)
    finally:
        session.close()
