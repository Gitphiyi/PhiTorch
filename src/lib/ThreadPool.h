#ifndef THREADPOOL_H
#define THREADPOOL_H
#include <pthread.h>

typedef void (*thread_func)(void *arg);

typedef struct {
    void* ret_val;
    pthread_cond_t  received_cond;
    pthread_mutex_t futures_mutex;
    int             done;
} future_t;

typedef struct {
    void (*fn)(void *);      // user callback
    void *args;
    struct job_t *next;
} job_t;
typedef struct {
    size_t          num_threads; //num of threads in pool
    pthread_t*      threads;
    job_t*          first_job;
    job_t*          last_job;
    pthread_mutex_t mutex;
    pthread_cond_t  job_ready_cond; //there is a job that needs to be processed
    int             stop; //boolean emergency stop for the thread pool
    size_t          working_threads; //number of current working threads
} thread_pool_t;

thread_pool_t*  tpool_create(size_t num_threads);
void            tpool_destroy(thread_pool_t *tp);

int             tpool_execute_job(thread_pool_t* tp);
int             tpool_add_job(thread_pool_t *tp, thread_func func, void *arg);
int             tpool_remove_job(thread_pool_t* tp);
void            tpool_wait(thread_pool_t *tp);
#endif