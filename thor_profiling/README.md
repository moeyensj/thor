# THOR Memory Profiling of Ray Workers

This directory contains the code to profile the memory usage of Ray workers. Because of the way they are initialized, you have to monkey patch using Python's sitecustomize framework.

1. Add the .pth file

```
cp ./thor_profiling/thor_profiling.pth /usr/local/lib/python3.10/dist-packages/thor_profiling.pth
```

2. Make sure you have memray and ray installed
3. Now when you run the cluster, it will automaticaly add memray profiling to the /tmp/ray/session_*/logs/worker* directories
