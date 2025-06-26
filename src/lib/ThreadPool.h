#ifndef THREADPOOL_H
#define THREADPOOL_H
#include <pthread.h>
typedef struct Job {
    void (*fn)(void *);      // user callback
    void *arg;
    struct Job *next;
} Job;

typedef struct {
    size_t NUM_THREADS;
    pthread_t* threads;
    pthread_mutex_t mutex;
    pthread_cond_t cv; //condition variables signal that the condition MIGHT have changed. thus the actual condition is usually in a while loop
    Job jobs_queue[256];
} ThreadPool;

void init();
void destroy();
void add_job();
void finish_job();
#endif