import importlib.util
from pathlib import Path


def test_detects_actual_python_code_launcher(tmp_path):
    file=Path(__file__).parents[1]/'scripts/finalize_b_after_training.py'
    spec=importlib.util.spec_from_file_location('b_finalizer',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    run=Path('/media/hdd/stud1/b_improved_taskaligned_20261004')
    proc=tmp_path/'123';proc.mkdir()
    # The actual launcher constructs root/run in Python, rather than passing
    # the combined path literally as a separate CLI argument.
    (proc/'cmdline').write_bytes(b"python\0-c\0root=Path('/media/hdd/stud1');train_b_improved(data,root/'b_improved_taskaligned_20261004')")
    assert module.trainer_alive(123,run,tmp_path)
    assert not module.trainer_alive(999,run,tmp_path)
    assert not module.trainer_alive(123,Path('/other/b_improved_taskaligned_20261004'),tmp_path)
