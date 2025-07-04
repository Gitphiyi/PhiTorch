#include "ThreadPool.h"
#include <pthread.h>

static Job* create_job(thread_func func_ptr, void* args) {
    if(func_ptr == NULL) {
        return NULL;
    }
    Job* job = malloc(sizeof(Job));
    job->args = args;
    job->fn = func_ptr;
    job->next = NULL;
    return job;
}
static void destroy_job(Job* job) {
    if(job == NULL) { return; }
    free(job);
}
static Job* get_job(ThreadPool* pool) {
    if(pool == NULL) { return NULL; }
    Job* job = pool->first_job;
    if(job->next == NULL) {
        pool->first_job = NULL;
        pool->last_job = NULL;
    } else {
        pool->first_job = job->next;
    }
    return job;
}

/*
 * get a job off of the pool and execute it
*/
void* pool_worker() { 
    // wait for a job to come through cv or stop on pool->stop
    // when a job comes poll it, increment working threads, and unlock mutex to allow other jobs to get added on
    // when job is done clean it up
    // lock mutex to decrement working threads, and signal for more work if there is no jobs on queue
    return NULL;
}