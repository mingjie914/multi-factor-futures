from __future__ import annotations

import numpy as np
import pandas as pd

from core.interfaces import Factor
from core.registry import register_factor


@register_factor("roll_yield_20d", category="term_structure")
class RollYield(Factor):
    """展期收益率因子 (近月-远月价差 / 近月价格).

    计算逻辑:
        1. 每个交易日取近月合约和远月合约的收盘价
        2. roll_yield = (far_close - near_close) / near_close
           正值 = 远月升水 (contango), 负值 = 近月升水 (backwardation)
        3. 对原始 roll_yield 做 20 日窗口均值平滑

    近月/远月确定 (避免未来函数):
        - 按合约 Wind 代码字符串排序 (等同于到期日排序)
        - 每个交易日取该日有数据的前两个合约
        - 退市日当天若仍有数据则计入, 次日自动切换

    换月跳空处理:
        - roll_yield 是比值, 换月时因对比合约变化会产生跳空
        - 这是真实市场信息 (反映了换月时的展期成本)
        - 不对价格做复权 (我们要的是真实的价差关系)
        - 20 日窗口均值会平滑跳空的影响

    注意: 不使用 cfuturescontractmapping 主力合约表, 因为:
        1. 主力合约表只给一个合约, 无法区分近月/远月
        2. 主力合约切换规则基于成交量, 可能含未来信息
    """

    name = "roll_yield_20d"
    category = "term_structure"
    frequency = "daily"
    description = "展期收益率 (近月-远月价差/近月价格, 20日均值平滑)"

    # 窗口大小
    ROLLING_WINDOW = 20

    def dependencies(self) -> list:
        # 通过 get_contract_pair 获取近月/远月价格, 不依赖标准 get(field) 接口
        return ["close"]

    def compute(self, data, dates, universe):
        # 获取近月和远月合约的收盘价
        pair = data.get_contract_pair("close", dates, universe)
        near_close = pair.get("near", pd.DataFrame())
        far_close = pair.get("far", pd.DataFrame())

        if near_close.empty or far_close.empty:
            return pd.DataFrame(np.nan, index=dates, columns=universe)

        # 对齐到 dates 和 universe
        near_close = near_close.reindex(index=dates, columns=universe)
        far_close = far_close.reindex(index=dates, columns=universe)

        # 计算展期收益率: (远月 - 近月) / 近月
        # 正值 = 远月升水 (contango), 负值 = 近月升水 (backwardation)
        # 避免除零: near_close <= 0 时置为 NaN
        with np.errstate(divide="ignore", invalid="ignore"):
            roll_yield = (far_close - near_close) / near_close
        roll_yield = roll_yield.where(near_close > 0, np.nan)

        # 20 日窗口均值平滑 (最小周期 10, 避免窗口初期全 NaN)
        result = roll_yield.rolling(self.ROLLING_WINDOW, min_periods=10).mean()

        return result
