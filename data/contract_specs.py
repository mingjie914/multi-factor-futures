"""合约规格表 (38品种) — 数据工程冻结导出 (2026-08-03, 2026-08-04 补保证金).

乘数来源: 股指/国债用 S_INFO_CEMULTIPLIER (元/点), 商品用 S_INFO_PUNIT (吨/手等).
保证金: S_INFO_FTMARGINS 解析 (合约价值比例).
TS(2年债)乘数 20000 是 T/TF/TL(10000) 的 2 倍; IF/IH=300, IC/IM=200, 手数换算务必注意.
"""

CONTRACT_SPECS = {
    "A": {"name": "黄大豆1号", "unit": "吨", "multiplier": 10.0, "quote": "人民币元/吨", "margin": 0.05},
    "AG": {"name": "白银", "unit": "千克", "multiplier": 15.0, "quote": "人民币元/千克", "margin": 0.04},
    "AL": {"name": "铝", "unit": "吨", "multiplier": 5.0, "quote": "人民币元/吨", "margin": 0.05},
    "AU": {"name": "黄金", "unit": "克", "multiplier": 1000.0, "quote": "人民币元/克", "margin": 0.04},
    "CU": {"name": "阴极铜", "unit": "吨", "multiplier": 5.0, "quote": "人民币元/吨", "margin": 0.05},
    "FU": {"name": "燃料油", "unit": "吨", "multiplier": 10.0, "quote": "人民币元/吨", "margin": 0.08},
    "HC": {"name": "热轧卷板", "unit": "吨", "multiplier": 10.0, "quote": "人民币元/吨", "margin": 0.04},
    "I": {"name": "铁矿石", "unit": "吨", "multiplier": 100.0, "quote": "人民币元/吨", "margin": 0.05},
    "IC": {"name": "中证500股指期货", "unit": "张", "multiplier": 200.0, "quote": "指数点", "margin": 0.08},
    "IF": {"name": "沪深300期货", "unit": "张", "multiplier": 300.0, "quote": "指数点", "margin": 0.08},
    "IH": {"name": "上证50股指期货", "unit": "张", "multiplier": 300.0, "quote": "指数点", "margin": 0.08},
    "J": {"name": "冶金焦炭", "unit": "吨", "multiplier": 100.0, "quote": "人民币元/吨", "margin": 0.05},
    "JM": {"name": "焦煤", "unit": "吨", "multiplier": 60.0, "quote": "人民币元/吨", "margin": 0.05},
    "M": {"name": "豆粕", "unit": "吨", "multiplier": 10.0, "quote": "人民币元/吨", "margin": 0.05},
    "MA": {"name": "甲醇", "unit": "吨", "multiplier": 10.0, "quote": "人民币元/吨", "margin": 0.05},
    "NI": {"name": "镍", "unit": "吨", "multiplier": 1.0, "quote": "人民币元/吨", "margin": 0.05},
    "P": {"name": "棕榈油", "unit": "吨", "multiplier": 10.0, "quote": "人民币元/吨", "margin": 0.05},
    "RB": {"name": "螺纹钢", "unit": "吨", "multiplier": 10.0, "quote": "人民币元/吨", "margin": 0.05},
    "RM": {"name": "菜籽粕", "unit": "吨", "multiplier": 10.0, "quote": "人民币元/吨", "margin": 0.05},
    "RU": {"name": "天然橡胶", "unit": "吨", "multiplier": 10.0, "quote": "人民币元/吨", "margin": 0.05},
    "SA": {"name": "纯碱", "unit": "吨", "multiplier": 20.0, "quote": "人民币元/吨", "margin": 0.05},
    "SN": {"name": "锡", "unit": "吨", "multiplier": 1.0, "quote": "人民币元/吨", "margin": 0.05},
    "SR": {"name": "白砂糖", "unit": "吨", "multiplier": 10.0, "quote": "人民币元/吨", "margin": 0.05},
    "T": {"name": "10年期国债期货", "unit": "张", "multiplier": 10000.0, "quote": "人民币元", "margin": 0.02},
    "TA": {"name": "精对苯二甲酸(PTA)", "unit": "吨", "multiplier": 5.0, "quote": "人民币元/吨", "margin": 0.05},
    "TL": {"name": "30年期国债期货", "unit": "张", "multiplier": 10000.0, "quote": "人民币元", "margin": 0.035},
    "TS": {"name": "2年期国债期货", "unit": "张", "multiplier": 20000.0, "quote": "人民币元", "margin": 0.005},
    "Y": {"name": "大豆原油", "unit": "吨", "multiplier": 10.0, "quote": "人民币元/吨", "margin": 0.05},
    "ZN": {"name": "锌", "unit": "吨", "multiplier": 5.0, "quote": "人民币元/吨", "margin": 0.05},
    "IM": {"name": "中证1000股指期货", "unit": "张", "multiplier": 200.0, "quote": "指数点", "margin": 0.08},
    "TF": {"name": "5年期国债期货", "unit": "张", "multiplier": 10000.0, "quote": "人民币元", "margin": 0.01},
    "CF": {"name": "一号棉花", "unit": "吨", "multiplier": 5.0, "quote": "人民币元/吨", "margin": 0.05},
    "OI": {"name": "菜籽油", "unit": "吨", "multiplier": 10.0, "quote": "人民币元/吨", "margin": 0.05},
    "LH": {"name": "生猪", "unit": "吨", "multiplier": 16.0, "quote": "人民币元/吨", "margin": 0.05},
    "JD": {"name": "鲜鸡蛋", "unit": "吨", "multiplier": 5.0, "quote": "人民币元/500千克", "margin": 0.05},
    "SC": {"name": "中质含硫原油", "unit": "桶", "multiplier": 1000.0, "quote": "人民币元/桶", "margin": 0.05},
    "V": {"name": "聚氯乙烯", "unit": "吨", "multiplier": 5.0, "quote": "人民币元/吨", "margin": 0.05},
    "UR": {"name": "尿素", "unit": "吨", "multiplier": 20.0, "quote": "人民币元/吨", "margin": 0.05},
}
