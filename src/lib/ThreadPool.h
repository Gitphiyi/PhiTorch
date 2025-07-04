#ifndef THREADPOOL_H
#define THREADPOOL_H
#include <pthread.h>

typedef void (*thread_func)(void *arg);

typedef struct Job {
    void (*fn)(void *);      // user callback
    void *args;
    struct Job *next;
} Job;

typedef struct {
    size_t num_threads; //num of threads in pool
    pthread_t* threads;
    Job* first_job;
    Job* last_job;
    pthread_mutex_t mutex;
    pthread_cond_t waiting_work_cond; //signals work needs to be processed
    pthread_cond_t inactive_pool_cond; //signals that all threads are not working
    int stop; 
    size_t working_threads; //number of current working threads
} ThreadPool;

void init();
void destroy();
void add_job();
void finish_job();

ThreadPool *tpool_create(size_t num);
void tpool_destroy(ThreadPool *tm);

int tpool_add_work(ThreadPool *tm, ThreadPool func, void *arg);
void tpool_wait(ThreadPool *tm);
#endif