"""一次性: 从 RDS cfuturescontpro 补保证金率, 更新 data/contract_specs.py.

S_INFO_FTMARGINS = '最低交易保证金:合约价值的X%' (兼容全角％). 追加 margin 字段.
"""
import sys
import re
import yaml
import pymysql

UNIV38 = ['A', 'AG', 'AL', 'AU', 'CU', 'FU', 'HC', 'I', 'IC', 'IF', 'IH', 'J', 'JM',
          'M', 'MA', 'NI', 'P', 'RB', 'RM', 'RU', 'SA', 'SN', 'SR', 'T', 'TA', 'TL',
          'TS', 'Y', 'ZN', 'IM', 'TF', 'CF', 'OI', 'LH', 'JD', 'SC', 'V', 'UR']

sys.path.insert(0, '.')
from data.contract_specs import CONTRACT_SPECS  # noqa: E402


def main():
    d = yaml.safe_load(open('config/local.yaml', encoding='utf-8'))
    m = d['data']['mysql']
    conn = pymysql.connect(host=m['host'], port=m['port'], user=m['user'],
                           password=m['password'], database=m['database'],
                           charset='utf8mb4', connect_timeout=10)
    cur = conn.cursor()
    ph = ','.join(['%s'] * len(UNIV38))
    cur.execute(
        "SELECT S_INFO_CODE, S_INFO_FTMARGINS FROM cfuturescontpro "
        f"WHERE S_INFO_CODE IN ({ph}) GROUP BY S_INFO_CODE", UNIV38)
    rows = {r[0]: r[1] for r in cur.fetchall()}

    missing = []
    for s in UNIV38:
        v = rows.get(s)
        m2 = re.search(r'(\d+(?:\.\d+)?)\s*[%％]', str(v))
        if not m2:
            missing.append(s)
            continue
        CONTRACT_SPECS[s]['margin'] = float(m2.group(1)) / 100.0

    if missing:
        print(f'WARNING: 保证金未解析 {missing}')

    # 重写 contract_specs.py (保留注释头)
    lines = [
        '"""合约规格表 (38品种) — 从 RDS cfuturescontpro 导出 (2026-08-03, 2026-08-04 补保证金).',
        '',
        '乘数来源: 股指/国债用 S_INFO_CEMULTIPLIER (元/点), 商品用 S_INFO_PUNIT (吨/手等).',
        '保证金: S_INFO_FTMARGINS 解析 (合约价值比例).',
        'TS(2年债)乘数 20000 是 T/TF/TL(10000) 的 2 倍; IF/IH=300, IC/IM=200, 手数换算务必注意.',
        '"""',
        '',
        'CONTRACT_SPECS = {',
    ]
    for s in UNIV38:
        r = CONTRACT_SPECS[s]
        lines.append('    "%s": {"name": "%s", "unit": "%s", "multiplier": %s, '
                     '"quote": "%s", "margin": %s},' % (
                         s, r['name'], r['unit'], r['multiplier'],
                         r.get('quote', ''), r.get('margin', 'None')))
    lines.append('}')
    open('data/contract_specs.py', 'w', encoding='utf-8').write('\n'.join(lines))
    print(f'已更新 data/contract_specs.py: {len(UNIV38)} 品种含保证金率')


if __name__ == '__main__':
    main()
