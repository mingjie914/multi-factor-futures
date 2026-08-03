"""一次性脚本: 从 RDS cfuturescontpro 导出 38 品种合约规格, 生成 data/contract_specs.py.

乘数来源: 股指/国债用 S_INFO_CEMULTIPLIER (元/点), 商品用 S_INFO_PUNIT (吨/手等).
运行后即可删除 (仅为落盘数据, 后续读取 data/contract_specs.py).
"""
import sys
import yaml
import pymysql

UNIV38 = ['A', 'AG', 'AL', 'AU', 'CU', 'FU', 'HC', 'I', 'IC', 'IF', 'IH', 'J', 'JM',
          'M', 'MA', 'NI', 'P', 'RB', 'RM', 'RU', 'SA', 'SN', 'SR', 'T', 'TA', 'TL',
          'TS', 'Y', 'ZN', 'IM', 'TF', 'CF', 'OI', 'LH', 'JD', 'SC', 'V', 'UR']


def main():
    d = yaml.safe_load(open('config/local.yaml', encoding='utf-8'))
    m = d['data']['mysql']
    conn = pymysql.connect(host=m['host'], port=m['port'], user=m['user'],
                           password=m['password'], database=m['database'],
                           charset='utf8mb4', connect_timeout=10)
    cur = conn.cursor()
    ph = ','.join(['%s'] * len(UNIV38))
    cur.execute(
        "SELECT S_INFO_CODE, S_INFO_NAME, S_INFO_TUNIT, S_INFO_PUNIT, "
        "S_INFO_CEMULTIPLIER, FS_INFO_PUNIT FROM cfuturescontpro "
        f"WHERE S_INFO_CODE IN ({ph}) GROUP BY S_INFO_CODE", UNIV38)
    rows = {r[0]: r for r in cur.fetchall()}
    missing = [s for s in UNIV38 if s not in rows]
    if missing:
        print(f'ERROR: 缺失品种 {missing}')
        sys.exit(1)

    lines = [
        '"""合约规格表 (38品种) — 从 RDS cfuturescontpro 导出 (2026-08-03).',
        '',
        '乘数来源: 股指/国债用 S_INFO_CEMULTIPLIER (元/点), 商品用 S_INFO_PUNIT (吨/手等).',
        'TS(2年债)乘数 20000 是 T/TF/TL(10000) 的 2 倍, 手数换算务必注意.',
        '"""',
        '',
        'CONTRACT_SPECS = {',
    ]
    for s in UNIV38:
        code, name, tunit, punit, cemul, fspunit = rows[s]
        mult = float(cemul) if cemul is not None else float(punit)
        line = ('    "%s": {"name": "%s", "unit": "%s", "multiplier": %s, "quote": "%s"},'
                % (s, name, tunit, mult, fspunit or ''))
        lines.append(line)
    lines.append('}')
    open('data/contract_specs.py', 'w', encoding='utf-8').write('\n'.join(lines))
    print(f'已写入 data/contract_specs.py, {len(UNIV38)} 品种')


if __name__ == '__main__':
    main()
