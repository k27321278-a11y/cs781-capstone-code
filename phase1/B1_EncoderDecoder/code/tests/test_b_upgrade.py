import json
from dataclasses import replace
import pytest
import torch


def test_word_windows_exact_deterministic_and_short_boundary():
    from cs781.b_upgrade import word_windows
    words=[f'w{i}' for i in range(1500)]
    rows=list(word_windows(words,'acts:1',781))
    assert rows == list(word_windows(words,'acts:1',781))
    assert rows and all(len(p)==512 and len(s)==128 and s[0]==words[start+512]
                        for start,p,s in rows)
    assert list(word_windows(words[:639],'x',781)) == []
    assert len(list(word_windows(words[:640],'x',781))) == 1


def test_task_aligned_keeps_heldout_and_tokenizer(catalog,tmp_path):
    from cs781.tokenizer import fit_tokenizer
    from cs781.prepare import prepare_data
    from cs781.b_upgrade import prepare_task_aligned
    tok=tmp_path/'tok.json'; fit_tokenizer(catalog,tok,512)
    base=prepare_data(catalog,tok,tmp_path/'base',781)
    new=prepare_task_aligned(base,catalog,tmp_path/'aligned',max_seq_len=32768)
    for split in ('dev','confirmation'):
        for suffix in ('jsonl','offsets.npy','prompts.json','references.json'):
            assert (base.root/f'{split}.{suffix}').read_bytes()==(new.root/f'{split}.{suffix}').read_bytes()
    assert new.tokenizer_sha256==base.tokenizer_sha256
    manifest=json.loads((new.root/'manifest.json').read_text())
    assert manifest['training_window_policy']['prompt_words']==512
    assert manifest['training_window_policy']['target_words']==128
    from cs781.tokenizer import load_tokenizer
    tokenizer=load_tokenizer(tok)
    for line in (new.root/'train.jsonl').read_text().splitlines():
        row=json.loads(line)
        assert len(tokenizer.decode(row['prefix_ids']).split())==512
        assert len(tokenizer.decode(row['suffix_ids']).split())==128
        assert row['document_id'].endswith(':0')


def test_selection_requires_full_finite_dev_and_actual_improvement():
    from cs781.b_upgrade import select_improved
    def result(score,ids=('a','b')):
        return {'checkpoint':'continued','policy':0,'report':{
            'declared_ids':list(ids),'metrics':{'bertscore_f1':score,'rouge_l_f1':.2,'bleu4':.01}}}
    best=select_improved([result(.82),result(.84)],{'a','b'},.81)
    assert best['report']['metrics']['bertscore_f1']==.84
    for rows in ([result(.79)],[result(float('nan'))],[result(.9,('a',))]):
        with pytest.raises(ValueError):select_improved(rows,{'a','b'},.81)


def test_extended_context_keeps_parameters_and_logits():
    from cs781.models.b import EncoderDecoderConfig,EncoderDecoderLegalModel
    cfg=EncoderDecoderConfig(vocab_size=32,max_seq_len=16,n_encoder_layers=1,
        n_decoder_layers=1,d_model=16,n_heads=2,d_head=8,d_ff=32,gradient_checkpointing=False)
    torch.manual_seed(1); a=EncoderDecoderLegalModel(cfg).eval()
    b=EncoderDecoderLegalModel(replace(cfg,max_seq_len=32)).eval()
    b.load_state_dict(a.state_dict(),strict=True)
    prompt=torch.tensor([[4,5,6]]); decoder=torch.tensor([[1,7]])
    assert a.count_parameters()==b.count_parameters()
    torch.testing.assert_close(a(prompt,decoder)[0],b(prompt,decoder)[0])


def test_declared_policies_cannot_silently_differ():
    from cs781.b_upgrade import validate_policies
    from pathlib import Path
    config=json.loads((Path(__file__).parents[1]/'configs/b_improved.json').read_text())
    validate_policies(config)
    config['decoding'][1]['repetition_penalty']=1.25
    with pytest.raises(ValueError,match='polic'):validate_policies(config)


def test_export_recovers_zip_without_context(tmp_path):
    from cs781.b_upgrade import finalize_export
    from cs781.submission import write_submission
    prompts={'a':' '.join(['prompt']*512)}
    predictions={'a':' '.join(['result']*128)}
    output=tmp_path/'submission.zip'
    write_submission(predictions,prompts,output)
    report=finalize_export(output,predictions,prompts,{'architecture':'B'})
    assert report['prediction_zip_sha256']
    assert output.with_suffix('.context.json').exists()


def test_continuation_initializes_own_weights_and_binds_lineage(catalog,tmp_path,monkeypatch):
    import cs781.train_b_improved as module
    from cs781.tokenizer import fit_tokenizer
    from cs781.prepare import prepare_data
    from cs781.models.b import EncoderDecoderConfig,EncoderDecoderLegalModel
    from cs781.contracts import TrainConfig
    tok=tmp_path/'tok.json';fit_tokenizer(catalog,tok,512)
    data=prepare_data(catalog,tok,tmp_path/'data',781,max_seq_len=128)
    cfg=EncoderDecoderConfig(vocab_size=512,max_seq_len=128,n_encoder_layers=1,
        n_decoder_layers=1,d_model=32,n_heads=4,d_head=8,d_ff=64,gradient_checkpointing=False)
    original=EncoderDecoderLegalModel(cfg).state_dict()
    lineage={'checkpoint_sha256':'a'*64,'state':{'target_tokens':1000}}
    monkeypatch.setattr(module,'initialization',lambda *args:(original,lineage))
    run=tmp_path/'continued'
    train=TrainConfig(target_tokens=100,micro_batch_size=1,grad_accum_steps=1,device='cpu')
    module.train_b_improved(data,cfg,train,run,tmp_path/'parent.pt',data,
                           stop_after_steps=0,smoke_test=True)
    saved=torch.load(run/'checkpoints/latest.pt',weights_only=False)
    assert saved['metadata']['initialization']==lineage
    assert saved['metadata']['variant']=='B_taskaligned'
    assert saved['state']['target_tokens']==0
    for key,value in original.items():torch.testing.assert_close(saved['model'][key],value)
    with pytest.raises(Exception,match='metadata mismatch'):
        module.train_b_improved(data,cfg,replace(train,target_tokens=101),run,tmp_path/'parent.pt',
                               data,resume=True,stop_after_steps=0,smoke_test=True)
