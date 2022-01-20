from rq import Queue, Connection
import os, time

# from ml.ml1 import run_ml
from ml.automl1 import run_ml


def main():
    n_jobs_test = [1, 2, 3, 4]
    q = Queue()

    async_results = {}
    for n_jobs in n_jobs_test:
        async_results[n_jobs] = q.enqueue(run_ml, n_jobs)

    start_time = time.time()
    done = False
    sum_cpu_time = 0
    case_done = []
    while not done:
        os.system('clear')
        print('Asynchronously: (now = %.2f)' % (time.time() - start_time,))
        done = True
        for x in n_jobs_test:
            result = async_results[x].return_value
            if result is None:
                done = False
                print(f"ML (n_jobs={x}) elapsed time: calculating")
            else:
                if x not in case_done:
                    sum_cpu_time += result
                    case_done.append(x)
                
                print(f"ML (n_jobs={x}) elapsed time: {result:.3f}")
        
        print('')
        print(f"Total CPU time of jobs: ({sum_cpu_time:.3f})")

        print('')
        print('To start the actual in the background, run a worker:')
        print('    python run_worker.py')
        time.sleep(0.2)

    print('Done')    


if __name__ == '__main__':
    # Tell RQ what Redis connection to use
    with Connection():
        main()