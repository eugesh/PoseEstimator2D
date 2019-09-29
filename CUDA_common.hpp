#pragma once

enum cudaTouchError {
             UnknownError=-1,
             Success=0,
             Stop=1,
             ErrorHaveCudaDevice=2,
             ErrorAllocateMemoryCPU=3,
             ErrorAllocateMemoryGPU=4,
             ErrorCopyMemoryGPU=5,
             ErrorLoadFile=6,
             ErrorWriteFile=7,
             ErrorOpenDir=8,
             ErrorMakeDir=9,
             ErrorCreateModel=10,
             ErrorCudaRun=11,
             ErrorInData=12,
             ErrorHaveFreeMemoryGPU=13,
             ErrorDivisionByZero=14
};
