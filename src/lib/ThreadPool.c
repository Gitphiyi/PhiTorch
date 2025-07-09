#include "ThreadPool.h"
#include <pthread.h>

static job_t* create_job(thread_func func, void* args) {
    if(func == NULL) {
        return NULL;
    }
    job_t* job = malloc(sizeof(job_t));
    job->args = args;
    job->fn = func;
    job->next = NULL;
    return job;
}
static void destroy_job(job_t* job) {
    if(job == NULL) { return; }
    free(job);
}
static job_t* get_job(thread_pool_t* pool) {
    if(pool == NULL) { return NULL; }
    job_t* job = pool->first_job;
    if(job->next == NULL) {
        pool->first_job = NULL;
        pool->last_job = NULL;
    } else {
        pool->first_job = job->next;
    }
    return job;
}

thread_pool_t* tpool_create(size_t num_threads) {
    if(num_threads <= 0) {
        return NULL;
    }
    thread_pool_t* tp = malloc(sizeof(thread_pool_t));
    tp->num_threads = num_threads;
    tp->working_threads = 0;
    tp->stop = 0;
    tp->threads = malloc(num_threads * sizeof(thread_pool_t));
    pthread_mutex_init(&tp->mutex, NULL);
    pthread_cond_init(&tp->job_ready_cond, NULL);
    tp->first_job = NULL;
    tp->last_job = NULL;
    for(size_t i = 0; i < tp->num_threads; i++) {
        int res = pthread_create(&tp->threads[i], NULL, tpool_execute_job, tp);
        if(res) {
            return NULL;
        }
        pthread_detach(tp->threads[i]);
    }
    return tp;
    //creates thread that runs tpool_execute_job()s
}
/*
 * get a job off of the pool and execute it. Return the function return
*/
void* tpool_execute_job(thread_pool_t* tp) { 
    job_t* running_job;
    while(1) {
        pthread_mutex_lock(&tp->mutex);
        while(tp->first_job == NULL && !tp->stop) {
            pthread_cond_wait(&tp->job_ready_cond, &tp->mutex);
        } // thread pool waiting for the first job to come in, but if emergency stop it will end loop
        if(tp->stop) //emergency stop
            break;

        running_job = get_job(tp);
        tp->working_threads++;
        pthread_mutex_unlock(&tp->mutex);
        if(running_job != NULL) {
            running_job->fn(running_job->args);
            tpool_remove_job(tp);
        }

    }
    pthread_mutex_lock(&tp->mutex);

    // wait for a job to come through cv or stop on pool->stop
    // when a job comes poll it, increment working threads, and unlock mutex to allow other jobs to get added on
    // when job is done clean it up
    // lock mutex to decrement working threads, and signal for more work if there is no jobs on queue
    pthread_mutex_unlock(&tp->mutex);
    return NULL;
}

int tpool_add_job(thread_pool_t* tp, thread_func func, void* args) {
    if(tp == NULL) {
        goto failed;
    }
    job_t* job_to_add = create_job(func, args);
    if(job_to_add == NULL) {
        goto failed;
    }
    pthread_mutex_lock(&tp->mutex);
    if(tp->first_job == NULL) {
        tp->first_job = job_to_add;
        tp->last_job = job_to_add;
    } else {
        tp->last_job->next = job_to_add;
        tp->last_job = job_to_add;
    }
    pthread_cond_broadcast(&tp->job_ready_cond);
    pthread_mutex_unlock(&tp->mutex);
    return 1;
    failed:
    return 0;
}

void tpool_wait(thread_pool_t* tp) {
    //pthread_cond_wait(&tp->waiting_work_cond, &tp->mutex);
}