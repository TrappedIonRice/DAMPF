import numpy as np
import multiprocessing

def worker(k):
    
    x = np.random.choice(np.arange(10))
    # print(x)
    
    return x



if __name__ == "__main__":

    N = multiprocessing.cpu_count()
    print(N)
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        
        result = pool.map(worker, range(N))
    
    s = np.zeros(10)
    for i in range(10):
        s[i] = result.count(i)
        
    print(s)
        
        

