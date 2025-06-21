#include "ThreadPool.h"
#include <pthread.h>


void init(pthread_cond_t* cv) {
    pthread_cond_init(cv, NULL);
}