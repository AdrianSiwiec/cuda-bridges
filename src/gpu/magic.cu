void magic() {
    void *tmpArray;
    cudaMalloc((void **)&(tmpArray), sizeof(int) * 1);
    return; 
}