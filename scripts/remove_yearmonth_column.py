# -*- coding: utf-8 -*-
"""移除 1d/15m/1m 表内嵌冗余 year_month 列 (根治 ArrowTypeError).

背景: 1d 表 273 个文件内嵌 year_month 列 (DuckDB PARTITION_BY 历史遗留),
与目录分区列同名, pyarrow 多文件合并时 dictionary/plain 编码冲突 →
ArrowTypeError: Field year_month has incompatible types.

修复: 备份原文件 → 读取并 drop year_month 列 → 原子重写 (数据列不变,
year_month 由目录分区提供, 与 5m 标准布局一致).

注意: 备份到 本地表/_backup_yearmonth_列移除_YYYYMMDD/ 保留原文件.
"""
import os
import shutil
import datetime
import glob
import pyarrow.parquet as pq
import pandas as pd

ROOTS = {
    '1d': r"E:\程明杰公司内容\期货行情数据\本地表\futureshistoryprices1d",
    '15m': r"E:\程明杰公司内容\期货行情数据\本地表\futureshistoryprices15m",
    '1m': r"E:\程明杰公司内容\期货行情数据\本地表\futureshistoryprices1m",
}
STAMP = datetime.date.today().strftime('%Y%m%d')


def affected_files(base):
    out = []
    for d in sorted(os.listdir(base)):
        if not d.startswith('year_month='):
            continue
        for f in glob.glob(os.path.join(base, d, '*.parquet')):
            try:
                schema = pq.read_schema(f)
                if 'year_month' in schema.names:
                    out.append(f)
            except Exception:
                pass
    return out


def main():
    all_affected = []
    for name, base in ROOTS.items():
        files = affected_files(base)
        all_affected.extend(files)
        print(f'{name}: {len(files)} 个文件含内嵌 year_month 列')
    if not all_affected:
        print('无受影响文件')
        return

    # 备份
    bak_root = os.path.join(os.path.dirname(ROOTS['1d']), f'_backup_yearmonth列移除_{STAMP}')
    os.makedirs(bak_root, exist_ok=True)
    print(f'备份目录: {bak_root}')

    # 逐文件处理: 备份 -> drop year_month -> 重写
    done = 0
    for f in all_affected:
        rel = os.path.relpath(f, os.path.dirname(ROOTS['1d']))  # 保持相对结构
        # 备份到同相对路径
        bak_path = os.path.join(bak_root, rel)
        os.makedirs(os.path.dirname(bak_path), exist_ok=True)
        shutil.copy2(f, bak_path)
        # 读取 -> drop -> 重写
        try:
            df = pd.read_parquet(f)
            if 'year_month' in df.columns:
                df = df.drop(columns=['year_month'])
            df.to_parquet(f, index=False)
            done += 1
        except Exception as e:
            print(f'  ❌ {f}: {str(e)[:80]} (已备份, 未修改)')
    print(f'完成: {done}/{len(all_affected)} 文件已移除 year_month 列并重写')


if __name__ == '__main__':
    main()
