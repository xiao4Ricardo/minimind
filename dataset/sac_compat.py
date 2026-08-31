"""Windows Smart App Control / WDAC 兼容层。

现象：SAC 处于强制模式时会拦截 pyarrow 未签名的 ``_dataset.cp311-win_amd64.pyd``
（事件日志 CodeIntegrity 3033/3077），使 ``import pyarrow.dataset`` 抛 ImportError，
进而让整个 ``import datasets`` 失败——即便文件本身完好、之前能正常运行。
SAC 没有单文件白名单，且一旦关闭必须重装系统才能重新开启，因此不走关策略的路。

可行性：``datasets`` 只在 ``folder_based_builder`` 的类型注解里用到 ``pyarrow.dataset``；
本项目实际走的 ``load_dataset('json', ...)`` 依赖的是 ``pyarrow.json``，未被拦截。
所以注入一个占位模块即可。占位类在**被真正调用时**才抛错，不会静默返回错误结果。

同时保留上游 issue #771 的用途：本模块须在 torch 之前导入，以固定 pyarrow/torch 的
DLL 加载顺序。未被拦截的机器上 ``_install_stub()`` 直接返回 False，行为完全不变。
"""
import sys
import types


def _install_stub():
    try:
        import pyarrow.dataset  # noqa: F401
        return False  # 未被拦截，保持原样
    except ImportError:
        pass

    import pyarrow

    _cache = {}

    def _placeholder(name):
        def _raise(*args, **kwargs):
            raise RuntimeError(
                f"pyarrow.dataset.{name} 被 Windows Smart App Control 拦截，无可用实现。"
                f"本项目的 load_dataset('json', ...) 不需要它；若你确实要用 Arrow Dataset "
                f"相关功能，需换一个能通过 SAC 的 pyarrow 构建。")
        return type(name, (), {"__init__": _raise, "__call__": _raise})

    def _getattr(name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _cache.setdefault(name, _placeholder(name))

    stub = types.ModuleType("pyarrow.dataset")
    stub.__getattr__ = _getattr
    sys.modules["pyarrow.dataset"] = stub
    pyarrow.dataset = stub
    for extra in ("pyarrow._dataset", "pyarrow._dataset_orc", "pyarrow._dataset_parquet"):
        sys.modules.setdefault(extra, types.ModuleType(extra))
    return True


STUBBED = _install_stub()

import datasets  # noqa: F401,E402  # 必须在 torch 之前完成（issue #771 的加载顺序 workaround）
