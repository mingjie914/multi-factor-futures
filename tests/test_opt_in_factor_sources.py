"""The default graph must work with candidate generators unavailable."""
import os
from pathlib import Path
import subprocess
import sys


def test_default_entrypoints_and_factor_compute_do_not_import_generators():
    code = r'''
import importlib.abc
import sys
class BlockGenerators(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == x or fullname.startswith(x + '.') for x in
               ('factor_mining', 'factors.specs', 'factors.spec_factor')):
            raise AssertionError('default flow imported generator: ' + fullname)
sys.meta_path.insert(0, BlockGenerators())
import main
import run_factor_workflow
import run_portfolio_workflow
import workflows.factor_adaptivity
import factors.library
from core.config import load_config
from core.registry import list_registered
from factors.engine import FactorEngine
from types import SimpleNamespace
import numpy as np
import pandas as pd
registry = list_registered('factor')['factor']
assert not any(n.startswith(('gp_daily_', 'mined_', 'external_daily__')) for n in registry)
dates = pd.bdate_range('2025-01-01', periods=60)
columns = ['A', 'B']
rng = np.random.default_rng(3)
frames = {f: pd.DataFrame(rng.uniform(100, 200, (60, 2)), index=dates, columns=columns)
          for f in ('close', 'volume')}
data = SimpleNamespace(frequency='daily', get=lambda field, d, u: frames[field].reindex(index=d, columns=u),
                       prefetch=lambda *a, **k: None)
engine = FactorEngine(data)
result = engine.compute_factors(['volume_price_corr_20d'], dates, columns)
assert result['volume_price_corr_20d'].notna().any().any()
config = load_config('config/default.yaml')
assert all(n in registry for n in config.factors)
from pipeline.runner import PipelineRunner
runner = PipelineRunner.__new__(PipelineRunner)
runner.config, runner.data_manager = config, data
runner._build_factor_layer()
pd.testing.assert_frame_equal(
    runner.factor_engine.compute_factors(['volume_price_corr_20d'], dates, columns)['volume_price_corr_20d'],
    result['volume_price_corr_20d'])
import json
from pathlib import Path
from core.config import load_strategy_library
from run_portfolio_workflow import _load_factor_definition
library = json.loads(Path(config.factor_library.path).read_text(encoding='utf-8'))
assert all(row['factor'] in registry for row in library['factors'])
catalog = load_strategy_library('config/strategy_library.yaml')
sets = {s.id: s.factors for s in catalog.factor_sets}
for strategy in catalog.strategies:
    if strategy.status == 'archived':
        continue
    if strategy.factor_set_id:
        names = sets[strategy.factor_set_id]
    elif strategy.factor_definition_path:
        names = _load_factor_definition(Path(strategy.factor_definition_path))['factors']
    else:
        names = load_config(strategy.config_path).factors
    assert all(n in registry for n in names), strategy.id
print('DEFAULT_GRAPH_ISOLATED')
'''
    environment = dict(os.environ, MF_MINED_CANDIDATE_SNAPSHOT='missing-stale-snapshot.json')
    result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                            env=environment, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'DEFAULT_GRAPH_ISOLATED' in result.stdout


def test_explicit_spec_catalog_does_not_start_gp_from_stale_environment():
    code = r'''
import importlib.abc
import sys
class NoGP(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'factor_mining' or fullname.startswith('factor_mining.'):
            raise AssertionError('SPEC catalog started GP: ' + fullname)
sys.meta_path.insert(0, NoGP())
from factors.library import load_research_factor_catalog
from core.registry import list_registered
load_research_factor_catalog()
before = dict(list_registered('factor')['factor'])
load_research_factor_catalog()
assert dict(list_registered('factor')['factor']) == before
assert 'return_5d_z' in before
assert not any(n.startswith(('gp_daily_', 'mined_')) for n in before)
'''
    environment = dict(os.environ, MF_MINED_CANDIDATE_SNAPSHOT='missing-stale-snapshot.json')
    result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                            env=environment, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
