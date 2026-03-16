from core.executor import BaseExecutor

class TorchExecutor(BaseExecutor):
    def __init__(self, spec, config):
        # 官方版本没有文件，传 None
        super().__init__("official_torch", None, spec, config)

    def compile(self):
        pass # 无需编译

    def run(self, inputs):
        # 🟢 借用 Spec 里的参考实现，这才是最权威的 Torch 性能
        self.spec.reference_impl(**inputs)
        return inputs[self.spec.output_name]