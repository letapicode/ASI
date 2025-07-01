from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PushBatchRequest(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[PushRequest]
    def __init__(self, items: _Optional[_Iterable[_Union[PushRequest, _Mapping]]] = ...) -> None: ...

class QueryBatchRequest(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[QueryRequest]
    def __init__(self, items: _Optional[_Iterable[_Union[QueryRequest, _Mapping]]] = ...) -> None: ...

class QueryBatchReply(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[QueryReply]
    def __init__(self, items: _Optional[_Iterable[_Union[QueryReply, _Mapping]]] = ...) -> None: ...

class PushRequest(_message.Message):
    __slots__ = ("vector", "metadata")
    VECTOR_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    vector: _containers.RepeatedScalarFieldContainer[float]
    metadata: str
    def __init__(self, vector: _Optional[_Iterable[float]] = ..., metadata: _Optional[str] = ...) -> None: ...

class PushReply(_message.Message):
    __slots__ = ("ok",)
    OK_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    def __init__(self, ok: bool = ...) -> None: ...

class QueryRequest(_message.Message):
    __slots__ = ("vector", "k")
    VECTOR_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    vector: _containers.RepeatedScalarFieldContainer[float]
    k: int
    def __init__(self, vector: _Optional[_Iterable[float]] = ..., k: _Optional[int] = ...) -> None: ...

class QueryReply(_message.Message):
    __slots__ = ("vectors", "metadata")
    VECTORS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    vectors: _containers.RepeatedScalarFieldContainer[float]
    metadata: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, vectors: _Optional[_Iterable[float]] = ..., metadata: _Optional[_Iterable[str]] = ...) -> None: ...
