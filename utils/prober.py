import inspect
import re

class KernelProber:
    @staticmethod
    def probe(solve_fn):
        has_kwargs = False
        try:
            # 尝试直接获取函数签名 主要是针对triton
            target_fn = solve_fn.fn if hasattr(solve_fn, 'fn') else solve_fn
            sig = inspect.signature(target_fn)
            kernel_params = set(sig.parameters.keys())
            has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        except Exception:
            # 尝试从docstring中解析 主要是针对cuda
            doc = solve_fn.__doc__ or ""
            match = re.search(r"solve\((.*?)\)", doc)
            kernel_params = set([p.split(':')[0].strip() for p in match.group(1).split(',') if p.strip()]) if match else set()
        
        return kernel_params, has_kwargs