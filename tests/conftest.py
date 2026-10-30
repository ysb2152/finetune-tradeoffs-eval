# src.data.sample 은 import 시점에 qwen·retrieve 모듈을 끌어오고 그 안에서 torch를
# 불러온다. CI는 모델을 돌리지 않고 순수 함수(clean_refs·paper_split·채점 등)만 빠르게
# 확인하므로, 무거운 모듈을 가짜 모듈로 바꿔 import만 통과시킨다. 테스트 대상 함수들은
# 이 가짜들을 실제로 호출하지 않는다.
import sys
import types

_qwen = types.ModuleType("src.generate.qwen")
_qwen.build_messages = lambda *args, **kwargs: (None, None)
_qwen.MODEL_NAME = "stub"
sys.modules["src.generate.qwen"] = _qwen

_pipeline = types.ModuleType("src.retrieve.pipeline")
_pipeline.within_paper = lambda *args, **kwargs: []
sys.modules["src.retrieve.pipeline"] = _pipeline
