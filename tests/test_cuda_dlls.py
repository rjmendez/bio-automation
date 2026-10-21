"""cuda_dlls.add_cuda_dll_dirs: existing CUDA/cuDNN folders go on the DLL path and PATH once; missing ones are returned, not hidden."""
from bio_automation import cuda_dlls


def fresh(monkeypatch):
    monkeypatch.setattr(cuda_dlls, "_added", set())


def test_existing_folders_are_added_to_the_dll_path_and_path(tmp_path, monkeypatch):
    fresh(monkeypatch)
    a, b = tmp_path / "cuda", tmp_path / "cudnn"
    a.mkdir()
    b.mkdir()
    calls, env = [], {"PATH": r"C:\Windows"}
    assert cuda_dlls.add_cuda_dll_dirs([a, b], add_dll_directory=calls.append, environ=env) == []
    assert calls == [str(a), str(b)]
    assert env["PATH"] == str(b) + ";" + str(a) + ";" + r"C:\Windows"


def test_a_missing_folder_is_returned_and_not_added(tmp_path, monkeypatch):
    fresh(monkeypatch)
    there, gone = tmp_path / "cuda", tmp_path / "gone"
    there.mkdir()
    calls, env = [], {"PATH": ""}
    assert cuda_dlls.add_cuda_dll_dirs([gone, there], add_dll_directory=calls.append, environ=env) == [gone]
    assert calls == [str(there)]


def test_a_second_call_adds_nothing_again(tmp_path, monkeypatch):
    fresh(monkeypatch)
    a = tmp_path / "cuda"
    a.mkdir()
    calls, env = [], {"PATH": "X"}
    cuda_dlls.add_cuda_dll_dirs([a], add_dll_directory=calls.append, environ=env)
    cuda_dlls.add_cuda_dll_dirs([a], add_dll_directory=calls.append, environ=env)
    assert calls == [str(a)]
    assert env["PATH"] == str(a) + ";X"
