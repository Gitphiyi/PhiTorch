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
    pthread_cond_t cv;
} ThreadPool;

void init();
void destroy();
void wait();
void signal();


#endif