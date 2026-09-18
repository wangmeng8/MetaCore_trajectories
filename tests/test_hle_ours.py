import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.hle_outcome_caa import MODELS, HOOK_COMMIT, build_identity, guard_resume, sha256, vector_metadata, verify_checkpoint
from scripts.hle_ours import ROOT, audit_smoke, evaluation_plan, serving_plan, vector_path
from scripts.prepare_hle_outcome_caa import request_body
from scripts.run_hle_with_tools import apply_cli_overrides, build_hle_with_tools_config, parse_args


@pytest.mark.parametrize('key', MODELS)
def test_original_ours_pt_and_row_selection(key):
    import torch
    from scripts.hle_ours_vectors import validate_ours_vector
    path = vector_path(key)
    source = json.loads(path.with_name('metadata.json').read_text(encoding='utf-8'))
    meta, digest, _ = validate_ours_vector(path, source, key, validate_pt=True)
    assert meta['shape'] == [MODELS[key][1][0]-1, MODELS[key][1][1]]
    cfg = json.loads(request_body(path, None, 1., {'OUTCOME_CAA_MODEL_KEY':key})['vllm_xargs']['steer'])
    assert cfg['tensor_key'] == 'steering_vector'
    assert cfg['vector_sha256'] == digest
    assert cfg['vector_layer'] == cfg['optimal_layer'] == MODELS[key][0]
    assert meta['model_fingerprint'] is None
    with pytest.raises(ValueError, match='Invalid layer'):
        request_body(path, MODELS[key][1][0]-1, 1., {})


def test_tamper_and_shifted_mapping_rejected(tmp_path):
    from scripts.hle_ours_vectors import validate_ours_vector
    src = vector_path('gptoss_20b')
    shutil.copytree(src.parent, tmp_path/'v')
    path = tmp_path/'v/steering_vector.pt'
    metadata = json.loads(path.with_name('metadata.json').read_text())
    changed = dict(metadata, source_layers=list(range(1,24)))
    with pytest.raises(ValueError, match='mapping'):
        validate_ours_vector(path, changed)
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='checksum'):
        vector_metadata(path)


@pytest.mark.parametrize('key', MODELS)
@pytest.mark.parametrize('mode', ['smoke', 'full'])
def test_plan_preserves_baseline_and_ignores_stale_env(key, mode, monkeypatch):
    monkeypatch.setenv('HLE_ALL_TASKS','0')
    monkeypatch.setenv('HLE_SYSTEM_PROMPT_FILE','old.txt')
    monkeypatch.setenv('HLE_MARK_INCORRECT_IDS','1,2')
    monkeypatch.setenv('HLE_TEMPERATURE','1.5')
    monkeypatch.setenv('OPENAI_BASE_URL','http://wrong/v1')
    command, env, rid = evaluation_plan(key,'v1',mode)
    cfg = build_hle_with_tools_config(apply_cli_overrides(env, parse_args(command[2:])))
    assert cfg.num_tasks == (3 if mode=='smoke' else None)
    assert cfg.model in ('gemma-4-31b','qwen3.6-27b','gpt-oss-20b')
    assert cfg.base_url != 'http://wrong/v1'
    assert cfg.system_prompt_file is None and not cfg.mark_incorrect_ids
    assert cfg.temperature is None and cfg.disable_scientific_search
    expected_workers = 16 if key == 'gemma4_31b_it' else 8
    assert (cfg.max_workers,cfg.max_completion_tokens,cfg.max_iterations)==(expected_workers,100000,15)
    assert '_ours_' in rid
    assert 'steer' not in json.loads(env['HLE_WITH_TOOLS_AUXILIARY_EXTRA_BODY_JSON']).get('vllm_xargs',{})


def test_gemma_custom_layer_and_alpha(tmp_path, monkeypatch):
    data = tmp_path / 'smoke.json'
    data.write_text(json.dumps([{'id': str(i)} for i in range(3)]))
    monkeypatch.setattr(
        'scripts.hle_ours.request_body',
        lambda path, layer, alpha, env: {
            'vllm_xargs': {'steer': json.dumps({
                'vector_layer': layer, 'optimal_layer': layer, 'coefficient': alpha
            })}
        },
    )
    command, env, rid = evaluation_plan(
        'gemma4_31b_it', 'l37_a03_v1', 'smoke',
        run_id='hle_gemma_ours_l37_a03_smoke3_v1', data_path=data,
        layer=37, alpha=0.3,
    )
    cfg = build_hle_with_tools_config(apply_cli_overrides(env, parse_args(command[2:])))
    steer = json.loads(json.loads(cfg.agent_extra_body_json)['vllm_xargs']['steer'])
    assert steer['vector_layer'] == steer['optimal_layer'] == 37
    assert steer['coefficient'] == 0.3
    assert '_l37_a03_' in rid


def test_identity_provenance_and_resume(tmp_path):
    key='gptoss_20b'; path=vector_path(key)
    meta, digest, mdigest=vector_metadata(path)
    service=dict(served_model_name='gpt',model_key=key, vector_sha256=digest,vector_metadata_sha256=mdigest,
                 deployment_reference_sha256=meta['deployment_reference_sha256'],table_method='REVEAL (Ours)',
                 prefix_caching=False,hook_commit=HOOK_COMMIT,worker_version='hle-outcome-caa-v2',
                 worker_sha256=sha256(ROOT/'patches/vllm_hook/steer_activation_worker.py'))
    sp=tmp_path/'service.json';sp.write_text(json.dumps(service))
    data=tmp_path/'data.json';data.write_text('[{"id":"1"}]')
    env={'HLE_MODEL':'gpt','HLE_DATA_PATH':str(data),'HLE_ALL_TASKS':'1','OUTCOME_CAA_SERVICE_MANIFEST':str(sp),
         'HLE_WITH_TOOLS_AGENT_EXTRA_BODY_JSON':json.dumps(request_body(path,12,1.,{}))}
    cfg=build_hle_with_tools_config(env)
    identity=build_identity(cfg,env)
    assert identity['table_method']=='REVEAL (Ours)' and identity['model_fingerprint'] is None
    assert identity['extraction_checkpoint_verified'] is False
    (tmp_path/'manifest.json').write_text(json.dumps({'outcome_caa':dict(identity,table_method='REVEAL-Base')}))
    with pytest.raises(ValueError, match='identity'):
        guard_resume(tmp_path,identity)
    sp.write_text(json.dumps(dict(service,vector_sha256='old')))
    with pytest.raises(ValueError, match='vector/cache'):
        build_identity(cfg,env)


def test_deployment_fingerprint_is_not_extraction_proof(tmp_path):
    cfg=tmp_path/'config.json';cfg.write_text('{}')
    meta={'method':'trajre_mvp','model_fingerprint':None,'deployment_reference_sha256':'ref',
          'deployment_reference':{'model_fingerprint':{'file_hashes':{'config.json':sha256(cfg)}}}}
    actual=verify_checkpoint(tmp_path,meta)
    assert actual['extraction_checkpoint_verified'] is False
    cfg.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='fingerprint'):
        verify_checkpoint(tmp_path,meta)


def test_smoke_audit_requires_fresh_vector_and_all_ranks(tmp_path):
    run=tmp_path/'run';raw=run/'raw/official_run';raw.mkdir(parents=True)
    identity={'table_method':'REVEAL (Ours)','dataset':{'task_count':3,'task_ids':['0','1','2']},'vector_sha256':'new','layer':12,'alpha':1.0}
    (run/'manifest.json').write_text(json.dumps({'outcome_caa':identity}))
    (raw/'hle_gpt.json').write_text(json.dumps({str(i):{'response':'answer'} for i in range(3)}))
    audit=tmp_path/'audit';audit.mkdir()
    events=[dict(event='last_prefix',request_id=str(i),rank=r,vector_sha256='new',layer=12,alpha=1.,applied=True,prefix_length=5,token_position=4) for i in range(3) for r in range(4)]
    path=audit/'worker-0.jsonl';path.write_text('\n'.join(map(json.dumps,events))+'\n')
    service={'audit_dir':str(audit),'tensor_parallel_size':4}
    audit_smoke(run,service,{})
    path.write_text('\n'.join(map(json.dumps,events[:-1]))+'\n')
    with pytest.raises(ValueError,match='rank'):
        audit_smoke(run,service,{})
    with pytest.raises(ValueError,match='rank'):
        audit_smoke(run,service,{path.name:path.stat().st_size})
