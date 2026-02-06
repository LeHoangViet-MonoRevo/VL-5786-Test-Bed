from enum import Enum


class ErrorCodeCommon(Enum):
    CMN_VAL_001 = 1
    CMN_VAL_002 = 2
    CMN_VAL_003 = 3
    CMN_VAL_008 = 8


class ErrorCodeCommonSystem(Enum):
    CMN_SYS_003 = 3


class ErrorCodeAISystem(Enum):
    RAI_SYS_001 = 1
    RAI_SYS_002 = 2
    RAI_SYS_003 = 3
    RAI_SYS_004 = 4


class ResponseStatus(Enum):
    UNKNOWN_STATUS = 0
    SUCCESS = 1
    FAILED = 2


class FileType(Enum):
    IMAGE = 1
    PDF = 2
