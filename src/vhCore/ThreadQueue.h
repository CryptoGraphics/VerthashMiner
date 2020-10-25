
/*
 * Copyright 2018-2020 CryptoGraphics
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version. See LICENSE for more details.
 */

#ifndef ThreadQueue_INCLUDE_ONCE
#define ThreadQueue_INCLUDE_ONCE

#include "List.h"
#include <stdlib.h>
#include <external/tinycthread/tinycthread.h>

//-----------------------------------------------------------------------------
// Thread queue entity
struct tq_ent
{
    void* data;
    struct list_head q_node;
};

//-----------------------------------------------------------------------------
// Thread queue
struct thread_q
{
    struct list_head q;

    bool frozen;

    mtx_t mutex;
    cnd_t cond;
};

//-----------------------------------------------------------------------------
// Thread queue API
inline static struct thread_q *tq_new(void);
inline static void tq_free(struct thread_q *tq);
inline static bool tq_push(struct thread_q *tq, void *data);
inline static void *tq_pop(struct thread_q *tq, const struct timespec *abstime);
inline static void tq_freezethaw(struct thread_q *tq, bool frozen);
inline static void tq_freeze(struct thread_q *tq);
inline static void tq_thaw(struct thread_q *tq);

//-----------------------------------------------------------------------------
struct thread_q *tq_new(void)
{
    struct thread_q *tq;

    tq = (struct thread_q*)calloc(1, sizeof(*tq));
    if (!tq)
        return NULL;

    INIT_LIST_HEAD(&tq->q);
    mtx_init(&tq->mutex, mtx_plain);
    cnd_init(&tq->cond);

    return tq;
}

//-----------------------------------------------------------------------------
void tq_free(struct thread_q *tq)
{
    struct tq_ent *ent, *iter;

    if (!tq)
    {
        return;
    }

    list_for_each_entry_safe(ent, iter, &tq->q, q_node, struct tq_ent)
    {
        list_del(&ent->q_node);
        free(ent);
    }

    cnd_destroy(&tq->cond);
    mtx_destroy(&tq->mutex);

    memset(tq, 0, sizeof(*tq)); /* poison */
    free(tq);
}

//-----------------------------------------------------------------------------
bool tq_push(struct thread_q *tq, void *data)
{
    struct tq_ent *ent;
    bool rc = true;

    ent = (struct tq_ent*)calloc(1, sizeof(*ent));
    if (!ent)
        return false;

    ent->data = data;
    INIT_LIST_HEAD(&ent->q_node);

    mtx_lock(&tq->mutex);

    if (!tq->frozen)
    {
        list_add_tail(&ent->q_node, &tq->q);
    } else {
        free(ent);
        rc = false;
    }

    cnd_signal(&tq->cond);
    mtx_unlock(&tq->mutex);

    return rc;
}

//-----------------------------------------------------------------------------
void *tq_pop(struct thread_q *tq, const struct timespec *abstime)
{
    struct tq_ent *ent;
    void *rval = NULL;
    int rc;

    mtx_lock(&tq->mutex);

    if (!list_empty(&tq->q))
        goto pop;

    if (abstime)
        rc = cnd_timedwait(&tq->cond, &tq->mutex, abstime);
    else
        rc = cnd_wait(&tq->cond, &tq->mutex);

    if (rc != thrd_success)
    {
        goto out;
    }

    if (list_empty(&tq->q))
    {
        goto out;
    }

pop:
    ent = list_entry(tq->q.next, struct tq_ent, q_node);
    rval = ent->data;

    list_del(&ent->q_node);
    free(ent);

out:
    mtx_unlock(&tq->mutex);

    return rval;
}

//-----------------------------------------------------------------------------
void tq_freezethaw(struct thread_q *tq, bool frozen)
{
    mtx_lock(&tq->mutex);

    tq->frozen = frozen;

    cnd_signal(&tq->cond);
    mtx_unlock(&tq->mutex);
}

//-----------------------------------------------------------------------------
void tq_freeze(struct thread_q *tq)
{
    tq_freezethaw(tq, true);
}

//-----------------------------------------------------------------------------
void tq_thaw(struct thread_q *tq)
{
    tq_freezethaw(tq, false);
}

//-----------------------------------------------------------------------------

#endif // !ThreadQueue_INCLUDE_ONCE
